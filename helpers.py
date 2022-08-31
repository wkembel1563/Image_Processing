import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import random
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
from scipy.spatial.distance import cdist

def modelError(model, s_matches, d_matches): 
    """Model Error

    computes the error for the transformation of source points
    into destination points

    Parameters
    ----------
    model: (3x3) Matrix
        the transformation model to be evaluated

    s_matches: (Mx2) list
        all keypoints matches from the source image

    d_matches: (Mx2) list
        all keypoints matches from the destination image

    Returns
    -------
    error: (double)
        the error between the points transformed 
    """
    error = 1

    # remember to divide by 'w' element after transformation 

    return error

def compute_affine_transform(in_points, out_points): 
    """COMPUTE AFFINE TRANSFORM

    creates an affine matrix to transform the input points into
    the output points

    Parameters
    ----------
    in_points: (s, 2) list
        list of s sample keypoints from the source image which match
        the samples from the destination image

    out_points: (s, 2) list
        matching list of s sample keypoints from the destination image 

    Returns
    -------
    model: (3x3) numpy matrix
        the affine transformation matrix that transforms the given input
        points into given output points
    """
    src = np.array(in_points)

    num_samples = src.shape[0]

    src_affine = np.concatenate((src, np.ones((num_samples, 1))), axis=1)
    zero_buffer = np.zeros_like(src_affine)
    r1 = np.concatenate((src_affine, zero_buffer), axis=1)
    r2 = np.concatenate((zero_buffer, src_affine), axis=1)
    A = np.empty((r1.shape[0] + r2.shape[0], r1.shape[1]), dtype=r1.dtype)
    A[0::2] = r1
    A[1::2] = r2

    dst = np.array(out_points)
    b = dst.ravel()

    # (A^T A)x = (A^T)b
    # x = [(A^T A)^-1] (A^T) b
    ATA_inv = np.linalg.pinv(A.T @ A)
    x = ATA_inv @ A.T @ b

    # convert parameters into a 3x3 affine matrix
    M = np.reshape(x, (2, 3))
    M = np.vstack((M, np.zeros((1, M.shape[1]))))
    M[2, 2] = 1

    return M


def compute_projective_transform(in_points, out_points): 
    """COMPUTE PROJECTIVE TRANSFORM

    creates a projective homography to transform the input points into
    the output points

    Parameters
    ----------
    in_points: (s, 2) list
        list of s sample keypoints from the source image which match
        the samples from the destination image

    out_points: (s, 2) list
        matching list of s sample keypoints from the destination image 

    Returns
    -------
    model: (3x3) numpy matrix
        the projective transformation matrix that transforms the given input
        points into given output points
    """

    # left half of projection matrix is equivalent to an affine matrix
    # A1 is the left half
    src = np.array(in_points)
    dst = np.array(out_points)

    num_samples = src.shape[0]

    src_affine = np.concatenate((src, np.ones((num_samples, 1))), axis=1)
    zero_buffer = np.zeros_like(src_affine)
    r1 = np.concatenate((src_affine, zero_buffer), axis=1)
    r2 = np.concatenate((zero_buffer, src_affine), axis=1)
    A1 = np.empty((r1.shape[0] + r2.shape[0], r1.shape[1]), dtype=r1.dtype)
    A1[0::2] = r1
    A1[1::2] = r2

    # add on the right half of the projection matrix
    # the full matrix is A
    row1 = []
    row2 = []
    for idx in range(len(src)):
        row1.append([-(src[idx][0] * dst[idx][0]), -(src[idx][1] * dst[idx][0])])
        row2.append([-(src[idx][0] * dst[idx][1]), -(src[idx][1] * dst[idx][1])]) 
    r12 = np.array(row1)
    r22 = np.array(row2)

    buffer = np.empty((A1.shape[0], 2), dtype=A1.dtype)
    buffer[0::2] = r12
    buffer[1::2] = r22
    A = np.concatenate((A1, buffer), axis=1)


    # solve for model parameeters
    # (A^T A)x = (A^T)b
    # x = [(A^T A)^-1] (A^T) b
    b = dst.ravel()
    ATA_inv = np.linalg.pinv(A.T @ A)
    x = ATA_inv @ A.T @ b

    # x = [a11, a12, ..., a32]
    # ADD a33 back in as 1
    x = np.append(x, [1], axis=0)

    # convert parameters into a 3x3 homography 
    M = np.reshape(x, (3, 3))

    return M


def fitModel(s_matches, d_matches, sample_idxs, mode): 
    """FIT MODEL
    
    Samples a set of matching points in the source and destination image 
    and creates an transformation matrix model to fit the points

    The model can be an affine or projective matrix depending on the 'mode'

    After creating the model. Its error is calculated to evaluate the fit.

    Parameters
    ----------
    s_matches: (M, 2) list
        M keypoints from source image. Defined by pixel locations. 

    d_matches: (M, 2) list
        M keypoints from destination image.

    sample_idxs: (1, S) list
        List of S indices used to sample the list of samples

    mode: (string)
        Range = {'affine', 'projective'}. determines which type of model 
        to fit to the data

    Returns
    -------
    fit_error: (double) 
        measure of the error generated by the model when applied to total set of points

    params: (1, 6) list
        list of parameters that define the affine matrix model
    """
    fit_error = 0.0
    params = []

    # collect sample points
    sample_in = [s_matches[i] for i in sample_idxs]
    sample_out = [d_matches[i] for i in sample_idxs]

    # build model to fit the sample points
    if mode == 'affine':
        model = compute_affine_transform(sample_in, sample_out)

    elif mode == 'projective':
        model = compute_projective_transform(sample_in, sample_out)

    ########################
    exit(1)

    # measure error
    fit_error = modelError(model, s_matches, d_matches) 

    # extract model parameters
    for val in model:
        params.append(val)

    return fit_error, params 


def ransac(s_matches, d_matches, N, min_samples, threshold, mode="affine"):
    """RANSAC

    finds a model that best fits the set of input keypoint matches

    Parameters
    ----------
    s_matches: (M, 2) array
        pixel locations (i,j) that correspond to keypoints in source image
        that matched destination keypoints

    d_matches: (M, 2) array
        keypoints in destination image that match those in the source image
        given by s_matches.

    N: (int)
        number of iterations to run the algorithm

    min_samples: (int)
        number of points to sample for each iteration. corresponds to the minimum
        number of points needed to fit the model

    threshold: ()
        the error threshold used to decide if a model fits a set of points

    mode: (string)
        model mode that decides if an affine model or a projective model will be used
        to fit the points. defaults to 'affine'

    Returns
    -------
        bestFit: 
            the model parameters that best fit the input data

    """

    fit_error = 0.0
    params = []
    sample_idxs = []
    best_fit_error = float('inf')
    best_fit_params = []

    # fit models N times
    for i in range(N):

        # randomly select 'min' num of sample poins
        max_match_idx = len(s_matches)
        for i in range(min_samples):
            sample_idxs.append(random.randint(0, max_match_idx)) 

        # fit points to an affine model
        fit_error, params = fitModel(s_matches, d_matches, sample_idxs, mode)

        ###########################
        print(fit_error)
        print(params)
        exit(1)

        # update best fit params
        if fit_error < threshold and fit_error < best_fit_error:
            best_fit_error = fit_error
            best_fit_params = params

    return params



def extractMatches(bf_matches, keypoints1, keypoints2): 
    """EXTRACT MATCHES

    uses the indices of matching points gotten from bute force matching to 
    find and return the set of matching keypoints in the source and dest
    images

    Parameters
    ----------
    bf_matches: (M, 2) array
        M indices (i, j) which show the ith keypoint in image 1 matched with the 
        jth keypoint in image 2

    keypoints1: (N, 2) array
        N pixel locations (x, y) from image 1 for each of its N keypoints

    keypoints2: (F, 2) array
        F pixel locations (x, y) from image 2 for each of its F keypoints


    Returns
    -------
    matches1: (M, 2) array
       pixel locations of the M keypoints of image 1 matched to M keypoints in image 2 

    matches2: (M, 2) array
       pixel locations of the M keypoints of image 2 matched to M keypoints in image 1 

    """
    matches1 = []
    matches2 = []

    # collect matches
    for match in bf_matches:
        matches1.append( list(keypoints1[match[0]]) ) 
        matches2.append( list(keypoints2[match[1]]) )

    return matches1, matches2


def getFeatures(img):
    """ GET FEATURES
    returns they keypoints and descriptors of a gray image

    parameters 
    -----------
    img: (M, N) array
        grayscale image with M rows of N pixels


    returns 
    --------
    keypoints: (M, 2) array
        pixel coordinates of M keypoints

    descriptors: (M, P) array
        descriptors arrays with P elements about M keypoints
        
    """
    # build extractor
    descriptor_extractor = SIFT()

    # extract features
    descriptor_extractor.detect_and_extract(img)

    # save features
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors

    return keypoints, descriptors


def matchFeatures(d1, d2):
    """MATCH FEATURES
    
    Brute force matching of image features done by calculating the L2 distance between 
    each pairing of descriptors in images 1 and 2. For pairing (i, j) the descriptor j 
    with the closest distance to descriptor i constitutes a match.
    
    parameters
    ----------
    d1: (M,N) array
        M descriptors (one for each keypoint) of size N for image 1
        
    d2: (M,N) array
        same as d1 but for image 2
    
    return
    ------
    match: (M, 2) array
        M indices (i, j) which show the ith keypoint in image 1 matched with the 
        jth keypoint in image 2
        
    """
    # constants
    num_of_descriptors = 0
    img2_match = 1
    
    # get (M,N) matrix of distances between d1 and d2 elements
    # element (i, j) of distances is the distance beween ith d1 descriptor
    #      and jth d2 descriptor 
    distances = cdist(d1, d2)
    
    # get a set of indices that represent each row in d1
    d1_idx = np.arange(d1.shape[num_of_descriptors])
    
    # get a set of indices that represent each matching row in d2
    # this is done by finding the col idx of the min distance for each row
    #      in the distances matrix. that col idx = row idx of matching d2 element
    d2_idx = np.argmin(distances, axis=img2_match)
    
    # pair idx of each d1 desciptor with the idx of it's matching d2 desciptor
    # i.e. each element (i, j) says the element i of d1 matches to element j of d2
    matches = np.column_stack((d1_idx, d2_idx))
    
    return matches
    

