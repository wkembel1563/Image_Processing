"""
Robert Kembel
1001337169
CSE 4310-001
Assignment 1 Task 3
"""

import sys
import numpy as np
import skimage.io as io
from img_transforms import resize_img
import matplotlib.pyplot as plt


##### CONSTANTS ######
IMGFILE = 1
PYRAMID_SIZE = 2
LENGTH = 0
WIDTH = 1
ARGS = 3

########
# PROGRAM FUNCTIONS
########
"""
VERIFY: Verify command line inputs are valid
"""
def verify(args, exp_num):
    # preparing relevant values
    arg_num = len(args)

    # check num of arguments
    if arg_num != exp_num:
        print("Verification Error. create_img_pyramid.py requires %d arguments to build pyramid" % (exp_num))
        exit(1)

    # pyramid size is positive int
    size = int(sys.argv[PYRAMID_SIZE])
    if size <= 0:
        print("Verification error. Pyramid size must be positive integer.")
        exit(1)

"""
PARSE_PATH: splits 'filename.ext' into 'filename' and 'ext'
returns two strings: filename and ext
"""
def parsePath(path):
    # find index of '.' in path
    idx = 0
    for i in range(len(path)): 
        if path[i] == '.':
            idx = i
            break

    # extract filename and extension
    filename = path[0:idx]
    ext = path[idx+1:]

    return filename, ext

"""
BUILD_PYRAMID: Resizes input image into p_size - 1 num of progressively smaller images
returns nth, but it saves
"""
def build_pyramid(img, p_size, filename, ext):
    # reduce img size by factor of 2 each iteration
    temp = img
    scale = 1
    for i in range(p_size - 1): 
        # resize
        temp = resize_img(temp, -2) 

        # save as <filename>_<scale>x.<ext>
        scale *= 2
        path = "%s_%dx.%s" % (filename, scale, ext)
        plt.imsave(path, temp)


########
# IMG PYRAMID LOGIC
########

"""
Verify Input
(arguments, expected arg num, verification mode)
"""
verify(sys.argv, ARGS)

"""
Extract arguments
"""
img_file_path = sys.argv[IMGFILE]
filename, ext = parsePath(img_file_path)

img = io.imread(img_file_path)
p_size = int(sys.argv[PYRAMID_SIZE])

"""
Build image pyramid
pyramid images will be saved in the function
"""
build_pyramid(img, p_size, filename, ext)
