from helpers import *

"""
get photos
"""
e1 = io.imread("IMAGES/eiffel1.jpg")
e2 = io.imread("IMAGES/eiffel2.jpg")

e1_gray = rgb2gray(e1)
e2_gray = rgb2gray(e2)

"""
find SIFT features, descriptors for both images
"""
keypoints1, descriptors1 = getFeatures(e1_gray)
keypoints2, descriptors2 = getFeatures(e2_gray)

"""
brute force matching
"""
bf_matches = matchFeatures(descriptors1, descriptors2)

"""
extract matching keypoints
"""
match1, match2 = extractMatches(bf_matches, keypoints1, keypoints2)

"""
build model via RANSAC
"""
iterations = 100
min_samples = 4
error_threshold = 5.0
model_type = "projective"

model_params = ransac(match1, match2, iterations, min_samples, error_threshold, model_type)
######################################
exit(1)

"""
use model to stitch images together
** remember to divide transformed points by 'w' to get final value
when using projective matrix
"""

