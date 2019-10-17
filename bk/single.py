import sys
import time
import cv2
import numpy as np
from scipy import signal
import astropy
from astropy.io import fits
import math
import random

#--------------------------------------------------------------

N = 1000

def load_images(fn):
        input_file = open(fn, "rb")

        images = []
        
        for frame_num in range(1):
                tmp = np.load(input_file)

                images.append(np.array(tmp))
        input_file.close()
        return images

#--------------------------------------------------------------


def calc_sum(images):
    sum = np.zeros((512,512))
    
    for frame_num in range(0,1):
        f1 = images[frame_num]
        sum = sum + f1

    return sum
 
  
#--------------------------------------------------------------

def dk_sum(images):
 
        sum = calc_sum(images)


        return sum/1.0
 
#--------------------------------------------------------------
# extract a single FITS from a bra file

def main(arg):

    fn = arg[1]
    print(fn)
    images = load_images(fn)
    sum = dk_sum(images)
        
    hdr = fits.header.Header()
    fits.writeto(fn + "_single_.fits", np.float32(sum), hdr, overwrite=True)

 
 #--------------------------------------------------------------
 

if __name__ == "__main__":
        main(sys.argv)

