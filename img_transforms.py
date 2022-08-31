"""
Robert Kembel
1001337169
CSE 4310-001
Assignment 1 Task 2
"""

import sys
import random 
import numpy as np
from numpy.lib import stride_tricks
import skimage.io as io
import matplotlib.pyplot as plt


##### CONSTANTS ######
IMGFILE = 1
CROP = 2
N = 2
SCALE = 2
LENGTH = 0
WIDTH = 1
MAX_HUE = 360
MAX_SAT = 1
MAX_VAL = 1 
MIN = 0
HUE = 2
SAT = 3
VAL = 4
RED = 0
GREEN = 1
BLUE = 2

########
# PROGRAM FUNCTIONS
########

"""
CROP: Crop a random square of the input image
parameters: rgb image, size of the crop in pixels
return cropped image of size 'size x size'
"""
def random_crop(img, size):
    img_length = img.shape[LENGTH]
    img_width = img.shape[WIDTH]

    # crop size must be positive 
    if size <= 0:
        print("Error. Crop size must be positive integer.")
        exit(1)

    # crop size should not exceed image size
    if size > img_length or size > img_width:
        print("Error. Crop size exceeds image size.")
        exit(1)

    crop = np.empty((size, size, 3))
    crop_row = 0
    crop_col = 0
    # offset of center from edge
    offset = size // 2

    # if the crop size is even, the center is a point
    if size % 2 == 0: 
        # get bounds of crop center
        vertical_bound = img_length - offset
        horiz_bound = img_width - offset

        # generate random centerpoint within bounds
        x = random.randint(offset, vertical_bound)
        y = random.randint(offset, horiz_bound)

        # get crop edges 
        begin_row = x - offset
        begin_col = y - offset
        end_row = x + offset   # actually need offset - 1, but +1 to prevent off-by-one error in for loop
        end_col = y + offset
        assert begin_row >= 0 and begin_col >= 0
        assert end_row <= img_length-1 and end_col <= img_width-1
        
        # crop image
        for ridx in range(begin_row, end_row):
            for cidx in range(begin_col, end_col):
                crop[crop_row, crop_col] = img[ridx, cidx]
                crop_col += 1
            crop_row += 1
            crop_col = 0


    # if the crop size is odd, the center is a pixel 
    else:
        # get bounds of crop center
        vertical_bound = (img_length - 1) - offset
        horiz_bound = (img_width - 1) - offset

        # generate random centerpoint within bounds
        x = random.randint(offset, vertical_bound)
        y = random.randint(offset, horiz_bound)

        # get crop edges
        begin_row = x - offset
        begin_col = y - offset
        end_row = x + offset + 1    # extra +1 to prevent off by one error in for loop
        end_col = y + offset + 1
        assert begin_row >= 0 and begin_col >= 0
        assert end_row <= img_length-1 and end_col <= img_width-1

        # crop image
        for ridx in range(begin_row, end_row):
            for cidx in range(begin_col, end_col):
                crop[crop_row, crop_col] = img[ridx, cidx]
                crop_col += 1
            crop_row += 1
            crop_col = 0
    
    # return crop to int format if converted to float
    crop = crop.astype(np.uint8)
    return crop

"""
EXTRACT_PATCH: Turns an SQUARE rgb image into n^2 nonoverlapping patched
parameters: rgb image, number of patches to cut it into
returns numpy array with the patches
"""
def extract_patch(img, num_patches): 
    # Get patch size (image is a square)
    H, W, D = img.shape
    size = H // num_patches

    # Shape of patches array
    shape = [H//size, W//size] + [size, size, 3]

    # Strides to navigate patches array
    # (nxt row patches, next patch, next row, next pixel)
    strides = [size * s for s in img.strides[0:2]] + list(img.strides)

    # build patches array 
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)

    return patches


"""
RESIZE_IMG: Resizes an image according to some scale factor using nearest neighbor interpolation
parameters: rgb image, scale factor (+ scale upsamples, - scale downsamples)
returns resized image as a numpy array
"""
def resize_img(img, factor):
    shp = img.shape

    # 0 or 1 scaling factor maps to no change
    if factor == 0 or factor == 1 or factor == -1: 
        return img

    # downsample image
    elif factor < 0: 
        factor *= -1
        if (shp[LENGTH] % factor) == 0 and (shp[WIDTH] % factor) == 0: 
            img_resized = np.zeros((shp[LENGTH] // factor, shp[WIDTH] // factor, 3))
        else:
            print("Error. Image size not divisible by factor '%d'. Unable to downsample." % (factor))
            exit(1)

        # sample top-left pixel of each factor x factor block to build downsampled image
        row_iterations = shp[LENGTH] // factor
        col_iterations = shp[WIDTH] // factor
        for i in range(row_iterations): 
            for j in range(col_iterations): 
                img_resized[i, j] = img[i * factor, j * factor]

    # upsample image
    else: 
        img_resized = np.zeros((shp[LENGTH] * factor, shp[WIDTH] * factor, 3))

        # use each pixel to interpolate a subsection of the new picture
        for ridx, row in enumerate(img):
            for cidx, pixel in enumerate(row):
                i = ridx * factor
                j = cidx * factor
                img_resized[i:i + factor, j:j + factor] = pixel

    return img_resized.astype(np.uint8)


"""
GETPIXELMAX: return the largest value in an rgb pixel
"""
def getPixelMax(r, g, b):
    if r >= g and r >= b:
        return r
    
    elif g >= r and g >= b:
        return g
    
    else:
        return b


"""
CALCSATURATION: return the saturation values for an rgb image
returns: numpy array with sat values
"""
def calcSaturation(rgb_array, V):
    S = np.empty(V.shape)
    
    # Chroma
    C = V - np.min(rgb_array, axis=2)
    
    # S = C / V
    for ridx, row in enumerate(V):
        for cidx, val in enumerate(row):
            if val == 0:
                S[ridx, cidx] = 0
            else:
                S[ridx, cidx] = C[ridx, cidx] / val
    return S


"""
CALCHUE: return the hue values for an rgb image
returns: numpy array with hue values
"""
def calcHue(rgb_array, V):
    # init
    redIdx = 0
    greenIdx = 1
    blueIdx = 2
    H = np.empty(V.shape)
    
    # Chroma
    C = V - np.min(rgb_array, axis=2)
    
    # Calculate hue for each pixel
    for ridx, row in enumerate(rgb_array):
        for pidx, pixel in enumerate(row):
            # Get values for hue calculation
            r = float(pixel[redIdx])
            g = float(pixel[greenIdx])
            b = float(pixel[blueIdx])
            c = C[ridx][pidx]
            v = getPixelMax(r, g, b)
            if v != V[ridx, pidx]:
                print("HueCalc: Invalid V-val")
                exit(1)

            # calculate hue
            if c == 0:
                H[ridx][pidx] = 0.0
            
            elif v == r:
                H[ridx][pidx] = (((g - b) / c) % 6)
            
            elif v == g:
                H[ridx][pidx] = (((b - r) / c) + 2)
            
            elif v == b:
                H[ridx][pidx] = (((r - g) / c) + 4)
            
            else: 
                print("Hue calculation error")
                exit(1)  
    H = H * 60
    return H

"""
RGBTOHSV: transform rgb image into hsv image 
returns: hsv image as a numpy array
"""
def rgbToHsv(array):
    
    # create float-type copy of image
    array_norm = np.copy(array)
    array_norm = array_norm.astype(float)

    # normalize pixels
    for ridx, row in enumerate(array_norm):
        for pidx, pixel in enumerate(row):
            sum = np.sum(pixel)
            if sum != 0:
                array_norm[ridx, pidx] = array_norm[ridx, pidx] / 255
    
    # VALUE 
    V = np.max(array_norm, axis=2)
    
    # SATURATION
    S = calcSaturation(array_norm, V)
    
    # HUE
    H = calcHue(array_norm, V)
                
    # Build HSV Array
    hsv = np.copy(array_norm)
    hsv[..., 0] = H
    hsv[..., 1] = S
    hsv[..., 2] = V
            
    return hsv

"""
CALCRGB: Perform intermediate calculation in HSV to RGB convertion 
return final RGB array 
"""
def calcRGB(h, x, c, m):

    # >> RGB
    image = np.empty((h.shape[0], h.shape[1], 3))
    for ridx, row in enumerate(h):
        for cidx, hue in enumerate(row):
            # temps for calculation 
            C = c[ridx, cidx]
            X = x[ridx, cidx]
            M = m[ridx, cidx]

            # assign rgb values
            if hue >= 0 and hue < 1:
                image[ridx, cidx, RED] = C + M
                image[ridx, cidx, GREEN] = X + M
                image[ridx, cidx, BLUE] = 0 + M

            elif hue >= 1 and hue < 2:
                image[ridx, cidx, RED] = X + M
                image[ridx, cidx, GREEN] = C + M
                image[ridx, cidx, BLUE] = 0 + M
                
            elif hue >= 2 and hue < 3:
                image[ridx, cidx, RED] = 0 + M
                image[ridx, cidx, GREEN] = C + M
                image[ridx, cidx, BLUE] = X + M

            elif hue >= 3 and hue < 4:
                image[ridx, cidx, RED] = 0 + M
                image[ridx, cidx, GREEN] = X + M
                image[ridx, cidx, BLUE] = C + M

            elif hue >= 4 and hue < 5:
                image[ridx, cidx, RED] = X + M
                image[ridx, cidx, GREEN] = 0 + M
                image[ridx, cidx, BLUE] = C + M

            elif hue >= 5 and hue < 6:
                image[ridx, cidx, RED] = C + M
                image[ridx, cidx, GREEN] = 0 + M
                image[ridx, cidx, BLUE] = X + M

    return image


"""
HSVTORGB: transform hsv image into rgb image 
returns: rgb image as numpy array 
"""
def hsvToRgb(hsv_img):

    # Get h, s, v components of image
    h, s, v = hsv_img[...,0], hsv_img[...,1], hsv_img[...,2] 

    # Chroma 
    c = s * v

    # Preprocessing 
    h = h / 60
    x = c * (1 - abs((h % 2) - 1))
    m = v - c

    # Get RGB image and reverse normalization
    rgb_img = calcRGB(h, x, c, m) 
    rgb_img[..., RED] *= 255
    rgb_img[..., GREEN] *= 255
    rgb_img[..., BLUE] *= 255
    rgb_img = rgb_img.astype(np.uint8)

    return rgb_img


"""
ADJUSTHSV: adjust hue, saturation and value according to user inputs
paramteres: hsv image, hue increment, sat increment, val increment
return adjusted hsv image as numpy array
"""
def adjustHsv(hsv_img, h, s, v):

    # Hue should be in [0 360]
    if h < MIN or h > MAX_HUE:
        print("Error. Hue must be in range [0, 360]")
        exit(1)

    # Saturation and value both are in [0, 1]
    if s < MIN or s > MAX_SAT:
        print("Error. Saturation must be in range [0, 1]")
        exit(1)

    if v < MIN or v > MAX_VAL:
        print("Error. Value must be in range [0, 1]")
        exit(1)

    # perturb values
    for ridx, row in enumerate(hsv_img):
        for cidx, pixel in enumerate(row):
            # change each val, make sure they are in range

            # hue increment
            hsv_img[ridx, cidx, 0] += h
            if hsv_img[ridx, cidx, 0] > MAX_HUE: 
                hsv_img[ridx, cidx, 0] = MAX_HUE

            # sat increment
            hsv_img[ridx, cidx, 1] += s
            if hsv_img[ridx, cidx, 1] > MAX_SAT: 
                hsv_img[ridx, cidx, 1] = MAX_SAT

            # val increment
            hsv_img[ridx, cidx, 2] += v
            if hsv_img[ridx, cidx, 2] > MAX_VAL: 
                hsv_img[ridx, cidx, 2] = MAX_VAL

    return hsv_img


"""
COLOR_JITTER: Randomly perturbs the HSV values of an image within range specified by input
parameters: rgb image, hue max, sat max, val max
returns image with altered hsv values
"""
def color_jitter(img, hue, saturation, value):
    # convert img to hsv 
    hsv = rgbToHsv(img)

    # get random perturbations
    h = random.randint(0, hue)
    s = random.uniform(0, saturation)
    v = random.uniform(0, value)

    # perturb values
    hsv = adjustHsv(hsv, h, s, v)

    # convert back to rgb 
    img_jittered = hsvToRgb(hsv)

    return img_jittered

