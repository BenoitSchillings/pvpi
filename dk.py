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
        
        for frame_num in range(N):
                tmp = np.load(input_file)

                images.append(np.array(tmp))
        input_file.close()
        return images

#--------------------------------------------------------------


def calc_sum(images):
    sum = np.zeros((512,512))
    
    for frame_num in range(0,N):
        f1 = images[frame_num]
        sum = sum + f1

    return sum
 
  
#--------------------------------------------------------------

def dk_sum(images):
 
        sum = calc_sum(images)


        return sum/1000.0
 
#--------------------------------------------------------------

from glob import glob
def main(arg):
        count = 0
        print(glob(arg[1]))
        vsum = np.zeros( (len(glob(arg[1])), 512,512))
        for fn in glob(arg[1])[0:]: 
                print(fn)
                images = load_images(fn)
                sum = dk_sum(images)
                del images
                count = count + 1.0
                vsum[int(count) - 1] = sum
        
        median = np.average(vsum, axis=0)
        hdr = fits.header.Header()
        fits.writeto("dk.fits", np.float32(median), hdr, overwrite=True)

 
 #--------------------------------------------------------------
 

if __name__ == "__main__":
        main(sys.argv)

