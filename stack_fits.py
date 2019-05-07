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
from skimage import color, data, restoration

sliders = [30, 1000, 20]


#--------------------------------------------------------------

MAG = 2

#--------------------------------------------------------------

def other_init():
        init = 1

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
        return(cv2.resize(image, (0,0), fx=MAG, fy=MAG, interpolation=cv2.INTER_CUBIC))

#--------------------------------------------------------------

# move the array by dx, dy pixels

def shift(array, dx, dy):
        result = cp.roll(cp.roll(array, int(dx), 0), int(dy), 1)

        return result

 #--------------------------------------------------------------

# move the array by dx, dy pixels

def shiftn(array, dx, dy):
        result = np.roll(np.roll(array, int(dx), 0), int(dy), 1)

        return result

#--------------------------------------------------------------

def error(array):  
        #return(-fit(array))
        
        #return(cp.max(array[XX-40:XX+40, YY-40:YY+40]))     
        return math.sqrt(cp.mean((array[80:910, 80:910])**2))

#--------------------------------------------------------------

BIAS = 466.0
GAIN = 110
tresh = [0.71, 1.89, 2.93, 3.95, 4.96, 5.97, 6.97, 7.98, 8.98, 9.98, 11, 12, 13, 14, 15]

YY = 426
XX = 355
#--------------------------------------------------------------

#optimal tresholding of images to avoid the sqrt(2) excess noise 
#per https://arxiv.org/pdf/astro-ph/0307305.pdf

def clip(array):
    tmp = cp.array(array.astype(float))
    #tmp = tmp - BIAS
    #print(tmp)
    #tmp = tmp.clip(0, 8000)
    tmp = tmp / GAIN

    tmp1 = cp.copy(tmp)

    if (True):
        for level in range(10, -1, -1):
                i, j = cp.where(tmp < tresh[level])
                tmp1[i, j] = level
    
    tmp1 = tmp1 * GAIN
    #print(tmp1)

    return cp.asnumpy(tmp1 * 10.0)

#--------------------------------------------------------------

N = 1000


def calc_sum(images, offsets):
    sum = cp.zeros((512*MAG,512*MAG))
    for frame_num in range(0,len(images)):
        f1 = images[frame_num]
        sum = sum + shift(f1, offsets[frame_num, 0], offsets[frame_num, 1])
        
    return sum
    
    
 #--------------------------------------------------------------
   
def calc_suma(images, offsets):
    
    sum = np.zeros((512*MAG,512*MAG))
    moved_images = np.ndarray((len(images), 512*MAG, 512*MAG))
    #print(moved_images.shape)
    
    for frame_num in range(0,len(images)):
        f1 = cp.asnumpy(images[frame_num])
        f1 = shiftn(f1, offsets[frame_num, 0], offsets[frame_num, 1])
        moved_images[frame_num] = f1
        sum = sum + f1
    
       
    minv = np.min(moved_images, axis=0)
    maxv = np.max(moved_images, axis=0)
    #print(minv)
    #print(sum.shape)
    return sum - minv - maxv
 

  
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

def load_images(fn):
        images = []
        for frame in fn:
                tmp = fits.getdata(frame, ext=0)
                print(np.mean(tmp))
                if (np.mean(tmp)>95000):
                        tmp = tmp / np.mean(tmp)
                        tmp = tmp * 10000.0
                        tmp = tmp * (100000.0/np.mean(tmp))
                        images.append(cp.array(tmp))
                #print(tmp)
        return images

#--------------------------------------------------------------


def opt(sum, images, offsets, out_fn):
        entropy = error(sum)                                    #initial quality measurement
        best_entropy = entropy
        plt.figure(figsize=(8,5))
        count = len(images)
        step = 15.5
        print("count is ", str(len(images)))
        for iter in range(2240000):
                idx = random.randint(0, (count-1))                  #choose a random candidate
                dx = np.random.normal(0, step)                   #choose a modification of the offset
                dy = np.random.normal(0, step)
            
                old_sum = cp.copy(sum)                          #save the current sum of images
            
                sum = sub_image(sum, images, offsets, idx)      #substract an image at old offset
 
                offset0 = offsets[idx, 0]
                offset1 = offsets[idx, 1]
                            
                offsets[idx, 0] += dx                           #modify offset
                offsets[idx, 1] += dy
            
                offsets[idx,0] = min(offsets[idx,0], 80)
                offsets[idx,1] = min(offsets[idx,1], 80)
                offsets[idx,0] = max(offsets[idx,0], -80)
                offsets[idx,1] = max(offsets[idx,1], -80)
                
                
                sum = add_image(sum, images, offsets, idx)      #add image at new offset
                new_entropy = error(sum)                        
            

                if ((new_entropy) <= best_entropy):             #is this worse
                        offsets[idx, 0] = offset0               #undo all
                        offsets[idx, 1] = offset1
                        sum = cp.copy(old_sum)
                        step = step * 0.99995
                if ((new_entropy) > best_entropy):              #score the new best entropy
                        #print(iter, new_entropy, step)
                        best_entropy = new_entropy
                        dist = math.sqrt(dx*dx+dy*dy)
                        #step = ((step * 10.0) + dist) / 11.0
                
               
                
                if (iter % 130 == 0):
                        suma = calc_suma(images, offsets)
                        cv2.imshow('suma', ((1.0/(sliders[1]+1.0)) * (cp.asnumpy(sum/3550.0) - sliders[0])))
                        print("entropy is " + str(best_entropy) + " " + str(step))
                        if cv2.waitKey(1) == 27:
                                break
 
                 
                    
                if (iter % 20000 == 0):                         #every N frames, recalc the whole stack to avoid
                        sum = calc_sum(images, offsets)         #creeping rounding errors
                        best_entropy = error(sum)
                        print("recal " + str(iter) + " " + str(step))
                        hdr = fits.header.Header()
                        suma = calc_suma(images, offsets)
                        fits.writeto(out_fn, np.float32(cp.asnumpy(suma)), hdr, overwrite=True)
                        #full = calc_suma(images, offsets)
                        #print(full)
                        #fits.writeto(out_fn, np.float32(cp.asnumpy(full)), hdr, overwrite=True)
                        #cv2.imshow('median', full/20000.0)
                        if cv2.waitKey(1) == 27:
                                break


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
 

        #hdr = fits.header.Header()
        #fits.writeto(out_fn, np.float32(cp.asnumpy(sum)), hdr, overwrite=True)


        return sum/20.0, offsets
 
#--------------------------------------------------------------

def main(arg):
        init_ui()
        other_init()

        print(arg[1:])
        images = load_images(arg[1:])
        sum, offsets = spec_sum(images[0], images, "result" + ".fits")
        del images
 
 
 #--------------------------------------------------------------
 

if __name__ == "__main__":
        main(sys.argv)

