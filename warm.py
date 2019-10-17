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
import argparse
import threading
import random

import skyx

import datetime

def debug_time():
    currentDT = datetime.datetime.now()
    print (str(currentDT))


#---------------------------------------------------------------------

FRAME_PER_FILE = 1000

#---------------------------------------------------------------------

def save_fits(nparray):
    hdr = fits.header.Header()
    fits.writeto('fn.fits', np.float32(nparray), hdr, overwrite=True)

#---------------------------------------------------------------------

def scale2(image):
    return(cv2.resize(image, (0,0), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST))

def scale3(image):
    return(cv2.resize(image, (0,0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST))

#---------------------------------------------------------------------

sliders = [300, 1000, 300, 1000]

clickpos = [256,256]

#---------------------------------------------------------------------

def set(idx, pos):
    sliders[idx] = pos + 1

#---------------------------------------------------------------------

def init_ui():
    cv2.namedWindow('live')
    cv2.namedWindow('sum')
    cv2.createTrackbar("Min", "live",0,16000, lambda pos: set(0, pos))
    cv2.setTrackbarPos("Min", "live", sliders[0]) 
    cv2.createTrackbar("Range", "live",11,63000, lambda pos: set(1, pos))
    cv2.setTrackbarPos("Range", "live", sliders[1]) 

    cv2.createTrackbar("Min", "sum",0,16000, lambda pos: set(2, pos))
    cv2.createTrackbar("Range", "sum",11,63000, lambda pos: set(3, pos))
    cv2.setTrackbarPos("Min", "sum", sliders[2]) 
    cv2.setTrackbarPos("Range", "sum", sliders[3]) 


#---------------------------------------------------------------------

class emccd:
    def __init__(self, temp):
        
        print("init cam")
        pvc.init_pvcam()

        
        self.vcam = next(Camera.detect_camera())
        self.vcam.open()
        self.vcam.gain=2
        print(self.vcam.temp)
        self.vcam.temp_setpoint = temp * 100 
        print(self.vcam.temp_setpoint)
        self.vcam.clear_mode="Pre-Sequence"
        #self.vcam.clear_mode="Pre-Exposure"

        pvc.set_param(self.vcam.handle, const.PARAM_READOUT_PORT, 0)
        #v = pvc.get_param(self.vcam.handle, const.PARAM_FAN_SPEED_SETPOINT, const.ATTR_CURRENT) 
        pvc.set_param(self.vcam.handle, const.PARAM_GAIN_MULT_FACTOR, 0)
        
        while(1):
            print(self.vcam.temp)
        

    def get_frame(self):        
        frame = self.vcam.get_live_frame().reshape(self.vcam.sensor_size[::-1]) 
        return frame
        
    def start(self, exposure):
        self.vcam.start_live(exp_time=int(exposure*1000.0))
        
    def close(self):
        self.vcam.close()


#---------------------------------------------------------------------


def main(args):

    cam_p = emccd(args.temp)

    exp_time_p = args.exp
    print(exp_time_p, args.filename)
    
    cam_p.start(exp_time_p)
    cnt = 1
    tot = 0
    saving = True
    sum = np.zeros((512,512))

    init_ui()

    if (args.filename == ''):
        saving = False
    if (saving):
        base_filename = args.filename + '_' + str(int(time.time())) + '_'
        save_p = saver(base_filename)
        
    while True:
        frame = cam_p.get_frame()
        f1 = frame.astype(float)

        sum = sum + f1
        vmax = np.max(f1)

        if (cnt % 2 == 0):
            cv2.imshow('live', scale2((1.0/sliders[1]) *  (f1 - sliders[0])))
            cv2.imshow('sum', scale2((1.0/sliders[3]) * (3.0*sum/cnt - sliders[2])))
        if (cnt % 5 == 0):
            curpos = cv2.minMaxLoc(cv2.GaussianBlur(f1,(3,3),0))[3]
            print(np.max(f1))
            if (curpos[0] > 30 and curpos[1] > 30 and curpos[0] < 480 and curpos[1] < 480):
                cv2.imshow('focus', scale3((1.0/sliders[1]) *  (f1[curpos[1]-30:curpos[1]+30, curpos[0]-30:curpos[0]+30] - sliders[0])))


        if (cnt % 5 == 0):
            if (clickpos[0] > 30 and clickpos[1] > 30 and clickpos[0] < 480 and clickpos[1] < 480):
                 cv2.imshow('click', scale3((1.0/sliders[3]) *  (sum[clickpos[1]-30:clickpos[1]+30, clickpos[0]-30:clickpos[0]+30]/cnt - sliders[2])))

        if (saving and (vmax > 850)):
            save_p.save_data(frame)
        
        guide_p.guide_sum(frame)
               
        if cnt == 10000:
            sum = np.zeros((512,512))
            cnt = 0
            

        key = cv2.waitKey(1)
        if key == 27:
            break
            
        if key == ' ':
            guide_p.move()
            
        

        cnt += 1
        tot += 1
        if (tot % 5000 == 0):
            guide_p.rand_pos()

    cam_p.close()
    save_p.close()
    
#---------------------------------------------------------------------

bg_active = True

def backgrounder(arg):
    while(True):
        print("bg" + str(bg_active), flush=True)
        time.sleep(1)

#---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-guide", "--temp", type=int, default = -70, help="target temp")
    args = parser.parse_args()
    #t = threading.Thread(target=backgrounder, args=(0,))
    #t.daemon = True
    #t.start()
    print(args)
    
    main(args)
     
