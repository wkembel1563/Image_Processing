# ImageProcessing

This is a suite of image processing scripts developed as a part of class projects for the University of Texas at Arlington's undergraduate computer vision course. The scripts provided perform color manipulation, image resizing to construct an image pyramid, and image stitching. The helper files also contain a function for cropping. 

## Main Scripts

change_hsv.py
- This script will increment the hue, saturation, and value of an input image by the input parameters
- The modified image will be saved as 'modified.jpg'
- Run: python3 change_hsv.py <image file> <hue> <sat> <val>

create_img_pyramid.py
- An 'image pyramid' will be created from the input image using this script
  - i.e. progressively smaller versions of the original image will be created depending on the pyramid size parameter
- Run: python3 create_img_pyramid.py <image file> <pyramid size> 

stitch_images.py
- This script produces a model capable of stitching two images together using SIFT features and the RANSAC algorithm
- It uses two images from the IMAGES folder to demonstrate

## Helper Files

img_transforms.py
- This contains helper functions used by create_img_pyramid.py

helpers.py
- This contains helper functions used by stitch_images.py
