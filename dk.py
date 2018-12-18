import sys
import time
import cv2
import numpy as np
import scipy
import scipy.optimize
from scipy import signal
import astropy
from astropy.io import fits
import scipy.ndimage as snd
import cupy as cp
import math
import random
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

sliders = [300, 1000, 0]


#--------------------------------------------------------------

def set(idx, pos):
        print(idx, pos)
        sliders[idx] = pos

#--------------------------------------------------------------

def click(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)


def init_ui():
        cv2.namedWindow('sum')

        cv2.createTrackbar("Min", "sum",0,16000, lambda pos: set(0, pos))
        cv2.createTrackbar("Range", "sum",0,16000, lambda pos: set(1, pos))
        cv2.setTrackbarPos("Min", "sum", sliders[0])
        cv2.setTrackbarPos("Range", "sum", sliders[1])
        cv2.setMouseCallback("sum", click)

#--------------------------------------------------------------

N = 1000

def load_images(fn):
        input_file = open(fn, "rb")

        images = []
        
        for frame_num in range(N):
                tmp = np.load(input_file)

                images.append(cp.array(tmp))
        input_file.close()
        return images

#--------------------------------------------------------------


def calc_sum(images):
    sum = cp.zeros((512,512))
    
    for frame_num in range(1,N):
        f1 = images[frame_num]
        sum = sum + f1

    return sum
 
  
#--------------------------------------------------------------

def dk_sum(images, out_fn):
 
        sum = calc_sum(images)


        return sum/1000.0
 
#--------------------------------------------------------------

def main(arg):
        init_ui()

        vsum = cp.zeros((512,512))
        count = 0
        print(arg[1:])
        for fn in arg[1:]: 
                images = load_images(fn)
                sum = dk_sum(images, fn + ".fits")
                del images
                count = count + 1.0
                vsum = vsum + sum
        
        hdr = fits.header.Header()
        fits.writeto(fn + ".fits", np.float32(cp.asnumpy(vsum / count)), hdr, overwrite=True)

 
 #--------------------------------------------------------------
 

if __name__ == "__main__":
        main(sys.argv)

