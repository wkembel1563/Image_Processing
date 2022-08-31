"""
Robert Kembel
1001337169
CSE 4310-001
Assignment 1 Task 1
"""

import sys
import numpy as np
import skimage.io as io
import skimage.color as color
import matplotlib.pyplot as plt

##### CONSTANTS ######
IMGFILE = 1
HUE = 2
SAT = 3
VAL = 4
MAX_HUE = 360
MAX_SAT = 1
MAX_VAL = 1 
LOWER_LMT = 0
HUE_UPPER_LMT = 360 
UPPER_LMT = 1
NUM_OF_ARGS = 5
RED = 0
GREEN = 1
BLUE = 2

########
# PROGRAM FUNCTIONS
########

"""
VERIFY: Verify command line inputs are valid
Format <filename> <imagefile> <hue> <sat> <val>
"""
def verify(args):
    # Should be 3 arguments + filename
    if len(args) != NUM_OF_ARGS:
        print("Error. change_hsv.py requires 4 arguments")
        exit(1)

    # Hue should be in [0 360]
    if int(args[HUE]) < LOWER_LMT or int(args[HUE]) > HUE_UPPER_LMT:
        print("Error. Hue must be in range [0, 360]")
        exit(1)

    # Saturation and value both are in [0, 1]
    if float(args[SAT]) < LOWER_LMT or float(args[SAT]) > UPPER_LMT:
        print("Error. Saturation must be in range [0, 1]")
        exit(1)
    if float(args[VAL]) < LOWER_LMT or float(args[VAL]) > UPPER_LMT:
        print("Error. Value must be in range [0, 1]")
        exit(1)


"""
PARSE INPUT: return each command line arg as its own object
"""
def parseInput(args):
    # Read image file
    filename = args[IMGFILE]
    img = io.imread(filename)

    return filename, img, float(args[HUE]), float(args[SAT]), float(args[VAL])


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
return adjusted hsv image
"""
def adjustHsv(hsv_img, h, s, v):
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


########
# PROGRAM LOGIC 
########

"""
Verify command line input
"""
verify(sys.argv)

"""
Parse input 
Should be of form: <filename> <HueMod> <SatMod> <ValMod> 
"""
file, img, hue, sat, val = parseInput(sys.argv)

"""
convert rgb image to hsv
"""
hsv_img = rgbToHsv(img)

"""
apply user adjustments
"""
hsv_img = adjustHsv(hsv_img, hue, sat, val)

"""
revert to rgb and save
"""
after_img = hsvToRgb(hsv_img)
plt.imsave("modified.jpg", after_img)
