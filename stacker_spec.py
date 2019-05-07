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
from scipy.ndimage import gaussian_filter


sliders = [300, 1000, 0]


#--------------------------------------------------------------

MAG = 2

#--------------------------------------------------------------

def other_init():
        global bias_frame
        
        bias_frame = fits.getdata("./bias.fits", ext=0)
        bias_frame = bias_frame.astype(float)
        print(bias_frame)


        global flat_frame
        
        flat_frame = fits.getdata("./flat.fits", ext=0)
        flat_frame = flat_frame.astype(float)
        flat_frame = flat_frame - bias_frame
        flat_frame /= np.max(flat_frame)
        print(flat_frame)
        
        
        global mask_frame
        global not_mask
        
        mask_frame = fits.getdata("./mask.fits", ext=0)
        mask_frame = flat_frame.astype(float) - 630.0
        mask_frame = np.clip(mask_frame, 0, 1.0)
        not_mask = 1.0 - mask_frame
        print(mask_frame)
       
        

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
        x = cv2.resize(image, (0,0), fx=MAG, fy=MAG, interpolation=cv2.INTER_NEAREST)
        x = x.astype(np.float32)
        #print(x.astype(np.float32))
        return x

#--------------------------------------------------------------

# move the array by dx, dy pixels

def shift(array, dx, dy):
        result = cp.roll(cp.roll(array, int(dx), 0), int(dy), 1)

        return result

#--------------------------------------------------------------

def fit(d):
        data = cp.asnumpy(d)
        loc = cv2.minMaxLoc(cv2.GaussianBlur(data[:, 0:400],(7,7),0))[3]

        z = cv2.resize(data[loc[1]-30:loc[1]+30, loc[0]-30:loc[0]+30], (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        
        y, x = np.mgrid[:60, :60]
        #g_init = models.Moffat2D(amplitude=z.max(), x_0 = 30, y_0=30)
        g_init = models.Gaussian2D(amplitude=z.max(), x_mean = 30, y_mean=30)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, x, y, z)
        #print(g.fwhm)

        #if (True):
            #plt.plot(x, fit_x, 'ko')
            #plt.plot(x, g(x), label='Gaussian')
            #plt.draw()
            #plt.pause(0.0001)
            #plt.clf()
        print("fwhm ", g.x_fwhm)
        return g.fwhm_x
 
#--------------------------------------------------------------

def error(array):  
        #return(-fit(array))
        
        #return(cp.max(array[XX-40:XX+40, YY-40:YY+40]))     
        return math.sqrt(cp.mean((array[80+80:990-80, 80+80:990-80]-100)**2))

#--------------------------------------------------------------

BIAS = 426.0
GAIN = 110
tresh = [0.71, 1.89, 2.93, 3.95, 4.96, 5.97, 6.97, 7.98, 8.98, 9.98, 11, 12, 13, 14, 15]

YY = 426
XX = 355
#--------------------------------------------------------------

#optimal tresholding of images to avoid the sqrt(2) excess noise 
#per https://arxiv.org/pdf/astro-ph/0307305.pdf

def clip(array):
    return array * 10

    tmp = cp.array(array.astype(float))
    #tmp = tmp - BIAS
    #return cp.asnumpy(tmp)
    #print(tmp)
    tmp = tmp.clip(0, 18000)
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

def gfwhm(data):
        loc = cv2.minMaxLoc(cv2.GaussianBlur(data[50:960, 50:960],(7,7),0))[3]
         
        z = cv2.resize(data[loc[1]+50-40:loc[1]+50+40, loc[0]+50-40:loc[0]+50+40], (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        
        y, x = np.mgrid[:80, :80]
        #g_init = models.Moffat2D(amplitude=z.max(), x_0 = 40, y_0=40)
        g_init = models.Gaussian2D(amplitude=z.max(), x_mean = 40, y_mean=40)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, x, y, z)

        return g.x_fwhm

#--------------------------------------------------------------

def chunk(data):
        loc = cv2.minMaxLoc(cv2.GaussianBlur(data[50:960, 50:960],(7,7),0))[3]

        z = cv2.resize(data[loc[1]+50-40:loc[1]+50+40, loc[0]+50-40:loc[0]+50+40], (0,0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        
        y, x = np.mgrid[:80, :80]
        #g_init = models.Moffat2D(amplitude=z.max(), x_0 = 40, y_0=40)
        g_init = models.Gaussian2D(amplitude=z.max(), x_mean = 40, y_mean=40)
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, x, y, z)
        #print(g.fwhm)

        #if (True):
            #plt.plot(x, fit_x, 'ko')
            #plt.plot(x, g(x), label='Gaussian')
            #plt.draw()
            #plt.pause(0.0001)
            #plt.clf()
            
        print("xy_fwhm ", g.x_fwhm)

        snr = data[80:100, 80:100]
        print(data.mean(), " ", data.max()," ", snr.std())
        return z

#--------------------------------------------------------------

def masker(data):
    #print("mask")
    #print(data)
    m1 = data * not_mask
    m2 = gaussian_filter(data, sigma=3)

    m2 = m2 * mask_frame
    
    #print(m1)
    #print(m2)
    return(m1+m2)


#--------------------------------------------------------------

def load_images(fn):
        input_file = open(fn, "rb")

        images = []
        
        try:
            for frame_num in range(N):
                    tmp = clip(np.load(input_file) - bias_frame)
                    tmp = masker(tmp)
                    tmp = tmp / flat_frame
                    tmp = scale2(tmp)
                    fw = gfwhm(tmp)
                    print(fw)
                    if (fw < 25.0):
                        images.append(cp.array(tmp))
        finally:             
            input_file.close()
            print("size is " + str(len(images)))
            return images

#--------------------------------------------------------------


def opt(sum, images, offsets, out_fn):
        entropy = error(sum)                                    #initial quality measurement
        best_entropy = entropy
        plt.figure(figsize=(8,5))
        count = len(images)
        step = 8
        for iter in range(300000):
                idx = random.randint(0, (count-1))                  #choose a random candidate
                dx = np.random.normal(0, step)                   #choose a modification of the offset
                dy = np.random.normal(0, step)
            
                old_sum = cp.copy(sum)                          #save the current sum of images
            
                sum = sub_image(sum, images, offsets, idx)      #substract an image at old offset

                offset0 = offsets[idx, 0]
                offset1 = offsets[idx, 1]
             
                offsets[idx, 0] += dx                           #modify offset
                offsets[idx, 1] += dy
            
                offsets[idx,0] = min(offsets[idx,0], 40)
                offsets[idx,1] = min(offsets[idx,1], 40)
                offsets[idx,0] = max(offsets[idx,0], -40)
                offsets[idx,1] = max(offsets[idx,1], -40)
            
                sum = add_image(sum, images, offsets, idx)      #add image at new offset
                new_entropy = error(sum)                        
            

                if ((new_entropy) <= best_entropy):              #is this worse
                        offsets[idx, 0] = offset0               #undo all
                        offsets[idx, 1] = offset1
                        sum = cp.copy(old_sum)
                        step = step * 0.999999
                        if (step < 2.0):
                            step = 2.0
                if ((new_entropy) > best_entropy):              #score the new best entropy
                        #print("entropy " + str(new_entropy))
                        best_entropy = new_entropy
                        dist = math.sqrt(dx*dx+dy*dy)
                        #step = ((step * 120.0) + (dist*1.5)) / 121.0
                 
               
                
                if (iter % 130 == 0):
                        cv2.imshow('sum', ((1.0/(sliders[1]+1.0)) * (cp.asnumpy(sum/3550.0) - sliders[0])))
                        if cv2.waitKey(1) == 27:
                                break
                if (iter % 1130 == 0):
                        cv2.imshow('chunk', ((1.0/(sliders[1]+1.0)) * (chunk(cp.asnumpy(sum/3550.0)) - sliders[0])))
                 
                    
                if (iter % 20000 == 0):                         #every N frames, recalc the whole stack to avoid
                        sum = calc_sum(images, offsets)         #creeping rounding errors
                        best_entropy = error(sum)
                        print("recal " + str(iter) + " " + str(step))
                        hdr = fits.header.Header()
                        fits.writeto(out_fn, np.float32(cp.asnumpy(sum)), hdr, overwrite=True)
                        
                        
    


#--------------------------------------------------------------

def spec_sum(model, images, out_fn):
 
# start with some random offsets

        offsets = np.random.rand(N,2)

# inital values between -4 and 4     

        offsets = offsets - 0.5
        offsets = offsets * 0.0
 
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
        for fn in arg[1:]: 
                print(fn)
                images = load_images(fn)
                sum, offsets = spec_sum(images[0], images, fn + ".fits")
                del images
 
 
 #--------------------------------------------------------------
 

if __name__ == "__main__":
        main(sys.argv)

