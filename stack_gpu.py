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

sliders = [300, 1000, 300, 1000]


def set(idx, pos):
        sliders[idx] = pos+1

def init_ui():
        cv2.namedWindow('live')
        cv2.namedWindow('sum')
        cv2.createTrackbar("Min", "live",0,16000, lambda pos: set(0, pos))
        cv2.setTrackbarPos("Min", "live", sliders[0])
        cv2.createTrackbar("Range", "live",1,16000, lambda pos: set(1, pos))
        cv2.setTrackbarPos("Range", "live", sliders[1])

        cv2.createTrackbar("Min", "sum",0,16000, lambda pos: set(2, pos))
        cv2.createTrackbar("Range", "sum",1,16000, lambda pos: set(3, pos))
        cv2.setTrackbarPos("Min", "sum", sliders[2])
        cv2.setTrackbarPos("Range", "sum", sliders[3])

def scale2(image):
        return(cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR))


def shift(array, dx, dy):
        result = cp.roll(cp.roll(array, int(dx), 0), int(dy), 1)

        return result

def error(array):
        error = cp.mean(array**2)
        return error

#def xy_trans(array, dx, dy):
        #result = shift(array, [dx, dy], mode='wrap', order=1)
        r#eturn result


def load_images(arg):
        input_file = open(arg[1], "rb")

        images = []
        
        for frame_num in range(10):
                tmp = np.load(input_file)
                tmp = scale2(tmp)
                print(cv2.minMaxLoc(cv2.GaussianBlur(tmp,(5,5),0))[3])

                images.append(cp.array(tmp) - 350.0)
        print(len(images))
        input_file.close()
        return images

def spec_sum(model, images):
        sum = images[0]
        offsets = np.zeros((10,2))
        for frame_num in range(1,10):
                f1 = images[frame_num]
                best_error = 1e20
            
                for dx in range(-25, 25, 3):
                        for dy in range(-25, 25, 3):
                                temp = shift(f1, dx, dy)

                                sumt = sum + temp
                                new_error = cp.mean((model - temp)**2)

                                if (new_error < best_error):
                                        best_dx = dx
                                        best_dy = dy
                                        best_error = new_error

                for dx in range(best_dx - 3, best_dx + 3, 1):
                        for dy in range(best_dy - 3, best_dy + 3, 1):
                                temp = shift(f1, dx, dy)

                                sumt = sum + temp
                                new_error = cp.mean((model - temp)**2)

                                if (new_error < best_error):
                                        best_dx = dx
                                        best_dy = dy
                                        best_error = new_error
                offsets[frame_num, 0] = best_dx
                offsets[frame_num, 1] = best_dy

                sum = sum + shift(f1, best_dx, best_dy)


        return sum, offsets
 


def main(arg):
        cnt = 1
        tot = 0
        fps = 0
        sum = cp.zeros((512,512))
        k = 0

        init_ui()

        images = load_images(arg)

        sum, offsets = spec_sum(images[0], images)
        print(offsets)
        
        s = cp.asnumpy(sum/10.0) 
        while(True):
            cv2.imshow('sum', ((1.0/sliders[3]) * (s - sliders[2])))

            if cv2.waitKey(1) == 27:
                break

  

if __name__ == "__main__":
        main(sys.argv)

