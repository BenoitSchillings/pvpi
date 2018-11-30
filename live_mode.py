from pyvcam import pvc
from pyvcam.camera import Camera
from pyvcam import constants as const

#---------------------------------------------------------------------

import sys
import time
import cv2
import numpy as np
import scipy
import astropy
from astropy.io import fits

#---------------------------------------------------------------------

FRAME_PER_FILE = 100

#---------------------------------------------------------------------

def save_fits(nparray):
    hdr = fits.header.Header()
    fits.writeto('fn.fits', np.float32(nparray), hdr, overwrite=True)

#---------------------------------------------------------------------

def scale2(image):
    return(cv2.resize(image, (0,0), fx=1.0, fy=1.0))

#---------------------------------------------------------------------

sliders = [300, 1000, 300, 1000]

#---------------------------------------------------------------------

def set(idx, pos):
    sliders[idx] = pos

#---------------------------------------------------------------------

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


#---------------------------------------------------------------------

class emccd:
    def __init__(self):
        pvc.init_pvcam()

        self.vcam = next(Camera.detect_camera())
        self.vcam.open()
        self.vcam.gain=3
        print(self.vcam.temp)
        self.vcam.temp_setpoint = -8000
        print(self.vcam.temp_setpoint)

        pvc.set_param(self.vcam.handle, const.PARAM_READOUT_PORT, 0)
        pvc.set_param(self.vcam.handle, const.PARAM_GAIN_MULT_FACTOR, 400)
        v = pvc.get_param(self.vcam.handle, const.PARAM_GAIN_MULT_FACTOR, const.ATTR_CURRENT)
        print(self.vcam.temp)
        

    def get_frame(self):        
        frame = self.vcam.get_live_frame().reshape(self.vcam.sensor_size[::-1]) 
        return frame
        
    def start(self, exposure):
        self.vcam.start_live(exp_time=int(exposure*1000.0))
        
    def close(self):
        self.vcam.close()


#---------------------------------------------------------------------


class saver:
    def __init__(self, filename):
        self.cnt = 0
        self.filename = filename
        self.idx = 0
        
    def save_data(self, data):
        self.cnt = self.cnt + 1
        if (self.cnt == 1):
            self.output_file = open(self.filename + str(self.idx), 'ab')
    
        np.save(self.output_file, data)
        if (self.cnt == FRAME_PER_FILE):
            self.output_file.close()
            self.cnt = 0
            self.idx = self.idx + 1
            
            
    def close(self):
        self.output_file.close()
        
        
#---------------------------------------------------------------------


def main(arg):

    camp = emccd()
    
    exp_time_p = float(arg[1])
    camp.start(exp_time_p)
    cnt = 1
    tot = 0
    saving = len(arg) > 2
    sum = np.zeros((512,512))

    init_ui()

    if (saving):
        base_filename = arg[2] + '_' + str(int(time.time())) + '_'
        savep = saver(base_filename)
        
    while True:
        frame = camp.get_frame()
        f1 = frame.astype(float)

        sum = sum + f1
        
        cv2.imshow('live', scale2((1.0/sliders[1]) *  (f1 - sliders[0])))
        cv2.imshow('sum', scale2((1.0/sliders[3]) * (sum/cnt - sliders[2])))
        
        if (saving):
            savep.save_data(frame)
                
        if cnt == FRAME_PER_FILE:
            print("file # " + str(seq) + " total frame = " + str(tot))
            sum = np.zeros((512,512))
            cnt = 0

        if cv2.waitKey(10) == 27:
            break

        cnt += 1
        tot += 1

    camp.close()
    savep.close()
    
#---------------------------------------------------------------------

if __name__ == "__main__":
    main(sys.argv)
