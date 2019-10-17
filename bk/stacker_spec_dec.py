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

sliders = [300, 1000, 0]

MAG = 1


#--------------------------------------------------------------

def set(idx, pos):
        print(idx, pos)
        sliders[idx] = pos

#--------------------------------------------------------------


def init_ui():
        cv2.namedWindow('sum')

        cv2.createTrackbar("Min", "sum",0,16000, lambda pos: set(0, pos))
        cv2.createTrackbar("Range", "sum",0,16000, lambda pos: set(1, pos))
        cv2.setTrackbarPos("Min", "sum", sliders[0])
        cv2.setTrackbarPos("Range", "sum", sliders[1])

#--------------------------------------------------------------

def scale2(image):
        return(cv2.resize(image, (0,0), fx=MAG, fy=MAG, interpolation=cv2.INTER_LINEAR))

#--------------------------------------------------------------

def shift(array, dx, dy):
        result = cp.roll(cp.roll(array, int(dx), 0), int(dy), 1)

        return result

#--------------------------------------------------------------

def error(array):
        return cp.std(array)

#--------------------------------------------------------------


def clip(array):
    #array = array.clip(500, 64000)
    return array

#--------------------------------------------------------------

N = 1000

def load_images(arg):
        input_file = open(arg[1], "rb")

        images = []
        
        for frame_num in range(N):
                tmp = clip(np.load(input_file))

                tmp = scale2(tmp)
                images.append(cp.array(tmp) - 250)
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
   
def calc_sum_i(images, offsets, included):
    sum = cp.zeros((512*MAG,512*MAG))
    
    for frame_num in range(1,N):
        f1 = images[frame_num]
        if (included[frame_num] != 0):
            sum = sum + shift(f1, offsets[frame_num, 0], offsets[frame_num, 1])

    return sum
  
    
def add_image(sum, images, offsets, frame_num):
        f1 = images[frame_num]
        sum = sum + shift(f1, offsets[frame_num, 0], offsets[frame_num, 1])
        return sum
        
def sub_image(sum, images, offsets, frame_num):
        f1 = images[frame_num]
        sum = sum - shift(f1, offsets[frame_num, 0], offsets[frame_num, 1])
        return sum

DIV = 2

def spec_sum(model, images, out_fn):
        offsets = np.random.rand(N,2)
        included = np.zeros(N)
        
        for i in range(int(N/DIV)):
            included[i] = 1
            
        offsets = offsets - 0.5
        offsets = offsets * 18.0
        #print(offsets)
 
        sum = calc_sum(images, offsets)
                
        entropy = error(sum)
        best_entropy = 0
        init_ui()
       
        
        for iter in range(70000):
            idx = random.randint(0, (N-1))
            dx = np.random.normal(0, 2.0)
            dy = np.random.normal(0, 2.0)
            
            old_sum = cp.copy(sum)
            
            sum = sub_image(sum, images, offsets, idx)
             
            offsets[idx, 0] += dx
            offsets[idx, 1] += dy
            
            
            sum = add_image(sum, images, offsets, idx)
            new_entropy = error(sum)
            

            if ((new_entropy) < best_entropy):
                offsets[idx, 0] -= dx
                offsets[idx, 1] -= dy
                sum = cp.copy(old_sum)
            if ((new_entropy) > best_entropy):
                print(iter, new_entropy, 4178.0)
                best_entropy = new_entropy
                
                
            if (iter % 130 == 0):
                cv2.imshow('sum', ((1.0/sliders[1]) * (cp.asnumpy(sum/1550.0) - sliders[0])))

                if cv2.waitKey(1) == 27:
                    break
                    
                    
            if (iter % 20000 == 0):
                sum = calc_sum(images, offsets)
                best_entropy = error(sum)
                print("recal " + str(iter))
                hdr = fits.header.Header()
                fits.writeto(out_fn, np.float32(cp.asnumpy(sum)), hdr, overwrite=True)

        sum0 = cp.asnumpy(sum)
              
        sum = calc_sum_i(images, offsets, included)
               
               
               
        best_entropy = error(sum * DIV)
        for iter in range(300000):
            idx1 = random.randint(0, (N-1))
            idx2 = random.randint(0, (N-1))
            if (iter % 330 == 0):
                print(iter, best_entropy)
                cv2.imshow('sumx', ((1.0/sliders[1]) * (cp.asnumpy(DIV*sum/1550.0) - sliders[0])))
                cv2.imshow('sum', ((1.0/sliders[1]) * (cp.asnumpy(sum0/1550.0) - sliders[0])))
                if cv2.waitKey(1) == 27:
                    break

            if (included[idx1] == 0 and included[idx2] == 1):
                included[idx1] = 1
                included[idx2] = 0
                sum = add_image(sum, images, offsets, idx1)
                sum = sub_image(sum, images, offsets, idx2)
                
                entropy = error(sum * DIV)
                
                if (entropy < best_entropy):
                    included[idx1] = 0
                    included[idx2] = 1
                    sum = sub_image(sum, images, offsets, idx1)
                    sum = add_image(sum, images, offsets, idx2)
                else:
                    best_entropy = entropy

        hdr = fits.header.Header()
        fits.writeto(out_fn, np.float32(cp.asnumpy(sum)), hdr, overwrite=True)


        return sum/20.0, offsets
 


def main(arg):
        cnt = 1
        tot = 0
        fps = 0
        sum = cp.zeros((512,512))
        k = 0

 
        images = load_images(arg)

        sum, offsets = spec_sum(images[0], images, arg[1] + ".fits")
        #init_ui()
        #while(True):
            #cv2.imshow('sum', ((1.0/sliders[1]) * (sum - sliders[2])))

            #if cv2.waitKey(1) == 27:
                #break

 
  

if __name__ == "__main__":
        main(sys.argv)

