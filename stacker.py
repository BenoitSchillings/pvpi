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

def save_fits(fn, nparray):
    hdr = fits.header.Header()
    fits.writeto(fn, np.float32(nparray), hdr, overwrite=True)



def scale2(image):
        return(cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def shift(array, dx, dy):
        result = np.roll(np.roll(array, dx, 0), dy, 1)
        return result

def error(array, loc):
        error = np.mean(array[loc[1] - 70:loc[1] + 70, loc[0] - 70:loc[0] + 70]**2)
        return error

def clip(array):
    a1 = array.clip(200, 1000)
    return a1

def main(arg):
        cnt = 1
        tot = 0
        fps = 0
        sum = np.zeros((512,512))
        sum = scale2(sum)
        k = 0


        print(len(arg))
        input_file = open(arg[1], "rb")

        model = np.load(input_file).astype(np.float32)
        image_data = fits.getdata("model.fits", ext=0)
        model = scale2(model)
        #model = image_data
        included = 0
        loc = cv2.minMaxLoc(cv2.GaussianBlur(model[:, :],(7,7),0))[3]
        for frame_num in range(999):
            frame = np.load(input_file)
            frame = scale2(frame) 
            f1 = frame.astype(np.float32)

            f1 = f1 - sliders[0]
            max = cv2.minMaxLoc(cv2.GaussianBlur(f1[:, :],(5,5),0))[1]
            print(frame_num, included, max)
            if (max < 2210):
                continue
                
                
            best_error = 1e20

            included = included + 1
            for dx in range(-25, 25, 3):
                for dy in range(-25, 25, 3):
                    temp = shift(frame, dx, dy)
                    sumt = model - temp
                    new_error = error(sumt, loc)
                    if (new_error < best_error):
                         best_dx = dx
                         best_dy = dy
                         best_error = new_error

            bdx = best_dx
            bdy = best_dy
            best_error = 1e20

            for dx in range(bdx-2, bdx+2, 1):
                for dy in range(bdy-2, bdy+2, 1):
                    temp = shift(frame, dx, dy)
                    #sumt = model - shift(frame, dx, dy)
                    sumt = model - temp
                    new_error = error(sumt, loc)
                    if (new_error < best_error):
                         best_dx = dx
                         best_dy = dy
                         best_error = new_error


            sum = sum + clip(shift(f1, best_dx, best_dy))
            print(best_dx, best_dy)

        s = (sum * 4.0) / included
        print(cv2.minMaxLoc(s[:, :])[1])

        save_fits('fn4.fits', s)
        init_ui()
        while(True):
            cv2.imshow('sum', ((1.0/sliders[3]) * (s - sliders[2])))

            if cv2.waitKey(1) == 27:
                break

        input_file.close()

if __name__ == "__main__":
        main(sys.argv)

