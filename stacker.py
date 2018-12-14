import sys
import time
import cv2
import numpy as np
import scipy
from scipy import signal
import astropy
from astropy.io import fits
import scipy.ndimage as snd

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
        return(cv2.resize(image, (0,0), fx=1, fy=1, interpolation=cv2.INTER_LINEAR))


def shift(array, dx, dy):
        result = np.roll(np.roll(array, dx, 0), dy, 1)
        return result

def error(array):
        error = np.mean(array**2)
        return error



def main(arg):
        cnt = 1
        tot = 0
        fps = 0
        sum = np.zeros((512,512))
        sum = scale2(sum)
        k = 0

        #init_ui()

        input_file = open(arg[1], "rb")

        model = np.load(input_file).astype(np.float32)
        model = scale2(model)
        
        for frame_num in range(1000):
            frame = np.load(input_file)
            frame = scale2(frame) 
            f1 = frame.astype(np.float32)

            f1 = f1 - sliders[0]

            best_error = 1e20
            print(' ')
            for dx in range(-13, 13, 1):
                print('.', end = "", flush = True)
                for dy in range(-13, 13, 1):
                    temp = shift(frame, dx, dy)
                    #sumt = model - shift(frame, dx, dy)
                    sumt = model - temp
                    new_error = error(sumt)
                    if (new_error < best_error):
                         best_dx = dx
                         best_dy = dy
                         best_error = new_error


            sum = sum + shift(f1, best_dx, best_dy)
            print(best_dx, best_dy)

        s = sum / 10.0
        
        init_ui()
        while(True):
            cv2.imshow('sum', ((1.0/sliders[3]) * (s - sliders[2])))

            if cv2.waitKey(1) == 27:
                break

        input_file.close()

if __name__ == "__main__":
        main(sys.argv)

