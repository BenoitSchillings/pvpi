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

MAG = 2


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

#--------------------------------------------------------------

def main(arg):
        init_ui()
        image = fits.getdata(arg[1], ext=0)
        image = image / np.mean(image)
        image = image * 1000

        while(True):
            cv2.imshow('sum', ((1.0/(sliders[1]+1.0)) * (image - sliders[0])))
            if cv2.waitKey(1) == 27:
                break

 
 
 #--------------------------------------------------------------
 

if __name__ == "__main__":
        main(sys.argv)

