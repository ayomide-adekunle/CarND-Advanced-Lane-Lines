import numpy as np
import cv2
import glob
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#Read in a calibration image
img = cv2.imread('../camera_cal/calibration1.jpg')

#Arrays to store object points and image points from all the images
objectpoints = [] #3D points in real world space
imgpoints = []    #2D points in hte image plane

# Get object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) #x, y coordinates

#Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

plt.imshow(img)
plt.show()






def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


