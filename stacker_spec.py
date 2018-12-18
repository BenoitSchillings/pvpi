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

# possible Pre-scaling input images for a pseudo drizzle

def scale2(image):
        return(cv2.resize(image, (0,0), fx=MAG, fy=MAG, interpolation=cv2.INTER_NEAREST))

#--------------------------------------------------------------

# move the array by dx, dy pixels

def shift(array, dx, dy):
        result = cp.roll(cp.roll(array, int(dx), 0), int(dy), 1)

        return result

#--------------------------------------------------------------

def fit(d):
        data = cp.asnumpy(d)
        loc = cv2.minMaxLoc(cv2.GaussianBlur(data,(7,7),0))[3]

        z = cv2.resize(data[loc[1]-30:loc[1]+30, loc[0]-30:loc[0]+30], (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        
        y, x = np.mgrid[:60, :60]
        g_init = models.Moffat2D(amplitude=z.max(), x_0 = 30, y_0=30)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, x, y, z)
        #print(g.fwhm)

        #if (True):
            #plt.plot(x, fit_x, 'ko')
            #plt.plot(x, g(x), label='Gaussian')
            #plt.draw()
            #plt.pause(0.0001)
            #plt.clf()
        print("fwhm ", g.fwhm)
        return g.fwhm
 
#--------------------------------------------------------------

def error(array):  
        #return(-fit(array))
        
        #return(cp.max(array[XX-40:XX+40, YY-40:YY+40]))     
        return math.sqrt(cp.mean((array[50:950, 50:950])**2))

#--------------------------------------------------------------

BIAS = 466.0
GAIN = 140
tresh = [0.71, 1.89, 2.93, 3.95, 4.96, 5.97, 6.97, 7.98, 8.98, 9.98, 11, 12, 13, 14, 15]

YY = 426
XX = 355
#--------------------------------------------------------------

#optimal tresholding of images to avoid the sqrt(2) excess noise 
#per https://arxiv.org/pdf/astro-ph/0307305.pdf

def clip(array):
    tmp = cp.array(array.astype(float))
    tmp = tmp - BIAS
    #print(tmp)
    #tmp = tmp.clip(0, 15000)
    tmp = tmp / 90.0

    tmp1 = cp.copy(tmp)

    #for level in range(10, -1, -1):
         #i, j = cp.where(tmp < tresh[level])
         #tmp1[i, j] = level
    
    tmp1 = tmp1 * GAIN
    #print(tmp1)

    return cp.asnumpy(tmp1 * 10.0)

#--------------------------------------------------------------

N = 1000

def load_images(fn):
        input_file = open(fn, "rb")

        images = []
        
        for frame_num in range(N):
                tmp = clip(np.load(input_file))

                tmp = scale2(tmp)
                images.append(cp.array(tmp))
        input_file.close()
        return images

#--------------------------------------------------------------


def calc_sum(images, offsets):
    sum = cp.zeros((512*MAG,512*MAG))
    
    for frame_num in range(1,N):
        f1 = images[frame_num]
        sum = sum + shift(f1, offsets[frame_num, 0], offsets[frame_num, 1])

    return sum
 
  
#--------------------------------------------------------------
    
def add_image(sum, images, offsets, frame_num):
        f1 = images[frame_num]
        sum = sum + shift(f1, offsets[frame_num, 0], offsets[frame_num, 1])
        return sum
  

#--------------------------------------------------------------
      
def sub_image(sum, images, offsets, frame_num):
        f1 = images[frame_num]
        sum = sum - shift(f1, offsets[frame_num, 0], offsets[frame_num, 1])
        return sum

#--------------------------------------------------------------

def chunk(data):
        loc = cv2.minMaxLoc(cv2.GaussianBlur(data,(7,7),0))[3]

        z = cv2.resize(data[loc[1]-30:loc[1]+30, loc[0]-30:loc[0]+30], (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        
        y, x = np.mgrid[:60, :60]
        g_init = models.Moffat2D(amplitude=z.max(), x_0 = 30, y_0=30)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, x, y, z)
        #print(g.fwhm)

        #if (True):
            #plt.plot(x, fit_x, 'ko')
            #plt.plot(x, g(x), label='Gaussian')
            #plt.draw()
            #plt.pause(0.0001)
            #plt.clf()
            
        print("xy_fwhm ", g.fwhm)
        return z

#--------------------------------------------------------------

def opt(sum, images, offsets, out_fn):
        entropy = error(sum)                                    #initial quality measurement
        best_entropy = entropy
        plt.figure(figsize=(8,5))
        for iter in range(52000000):
                idx = random.randint(0, (N-1))                  #choose a random candidate
                dx = np.random.normal(0, 1.5)                   #choose a modification of the offset
                dy = np.random.normal(0, 1.5)
            
                old_sum = cp.copy(sum)                          #save the current sum of images
            
                sum = sub_image(sum, images, offsets, idx)      #substract an image at old offset
             
                offsets[idx, 0] += dx                           #modify offset
                offsets[idx, 1] += dy
            
            
                sum = add_image(sum, images, offsets, idx)      #add image at new offset
                new_entropy = error(sum)                        
            

                if ((new_entropy) < best_entropy):              #is this worse
                        offsets[idx, 0] -= dx                   #undo all
                        offsets[idx, 1] -= dy
                        sum = cp.copy(old_sum)
                if ((new_entropy) > best_entropy):              #score the new best entropy
                        #print(iter, new_entropy, 4178.0)
                        best_entropy = new_entropy
                
                
                if (iter % 130 == 0):
                        cv2.imshow('sum', ((1.0/sliders[1]) * (cp.asnumpy(sum/3550.0) - sliders[0])))
                        cv2.imshow('chunk', ((1.0/sliders[1]) * (chunk(cp.asnumpy(sum/3550.0)) - sliders[0])))
                        if cv2.waitKey(1) == 27:
                                break
                    
                    
                if (iter % 20000 == 0):                         #every N frames, recalc the whole stack to avoid
                        sum = calc_sum(images, offsets)         #creeping rounding errors
                        best_entropy = error(sum)
                        print("recal " + str(iter))
                        hdr = fits.header.Header()
                        fits.writeto(out_fn, np.float32(cp.asnumpy(sum)), hdr, overwrite=True)


#--------------------------------------------------------------

def spec_sum(model, images, out_fn):
 
# start with some random offsets

        offsets = np.random.rand(N,2)

# inital values between -4 and 4     

        offsets = offsets - 0.5
        offsets = offsets * 3.0
 
#initial sum images

        sum = calc_sum(images, offsets)
                
        
        opt(sum, images, offsets, out_fn)
 

        hdr = fits.header.Header()
        fits.writeto(out_fn, np.float32(cp.asnumpy(sum)), hdr, overwrite=True)


        return sum/20.0, offsets
 
#--------------------------------------------------------------

def main(arg):
        init_ui()

        print(arg[1:])
        for fn in arg[1:]: 
                images = load_images(fn)
                sum, offsets = spec_sum(images[0], images, fn + ".fits")
                del images
 
 
 #--------------------------------------------------------------
 

if __name__ == "__main__":
        main(sys.argv)

