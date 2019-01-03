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

import skyx

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

#---------------------------------------------------------------------

def set(idx, pos):
    sliders[idx] = pos

#---------------------------------------------------------------------

def init_ui():
    cv2.namedWindow('live')
    cv2.namedWindow('sum')
    cv2.createTrackbar("Min", "live",0,16000, lambda pos: set(0, pos))
    cv2.setTrackbarPos("Min", "live", sliders[0]) 
    cv2.createTrackbar("Range", "live",11,16000, lambda pos: set(1, pos))
    cv2.setTrackbarPos("Range", "live", sliders[1]) 

    cv2.createTrackbar("Min", "sum",0,16000, lambda pos: set(2, pos))
    cv2.createTrackbar("Range", "sum",11,16000, lambda pos: set(3, pos))
    cv2.setTrackbarPos("Min", "sum", sliders[2]) 
    cv2.setTrackbarPos("Range", "sum", sliders[3]) 


#---------------------------------------------------------------------

class emccd:
    def __init__(self, gain):
        pvc.init_pvcam()

        self.vcam = next(Camera.detect_camera())
        self.vcam.open()
        self.vcam.gain=3
        print(self.vcam.temp)
        self.vcam.temp_setpoint = -8000
        print(self.vcam.temp_setpoint)

        pvc.set_param(self.vcam.handle, const.PARAM_READOUT_PORT, 0)
        print("gain = ", gain)
        pvc.set_param(self.vcam.handle, const.PARAM_GAIN_MULT_FACTOR, gain)
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
        if (self.filename == ""):
            return
            
        self.cnt = self.cnt + 1
        if (self.cnt == 1):
            self.output_file = open(self.filename + str(self.idx), 'ab')
    
        np.save(self.output_file, data)
        if (self.cnt == FRAME_PER_FILE):
            self.output_file.close()
            self.cnt = 0
            self.idx = self.idx + 1
            
            
    def close(self):
        if (self_filename != ""):
            self.output_file.close()
        
        
#---------------------------------------------------------------------

class guider:
    def __init__(self, frame_per_guide):
        self.frame_per_guide = frame_per_guide
        if (frame_per_guide != 0):
            self.sky = skyx.sky6RASCOMTele()
            self.sky.Connect()
            print(self.sky.GetRaDec())
            #self.sky.bump(0.5, 0.5)
            self.inited = False
            self.tracks_x = np.zeros((self.frame_per_guide))
            self.tracks_y = np.zeros((self.frame_per_guide))
            self.idx = 0
            self.initpos = [0,0]
            sum = np.zeros((512,512))
        
        
    def guide(self, image):
        if (self.frame_per_guide == 0):
            return
        
        self.curpos = cv2.minMaxLoc(cv2.GaussianBlur(image,(5,5),0))[3]
        
        
        self.tracks_x[self.idx] = self.curpos[0]
        self.tracks_y[self.idx] = self.curpos[1]
            
        self.idx = self.idx + 1
        
        if (self.idx == self.frame_per_guide):
            self.idx = 0
            mx = np.median(self.tracks_x)
            my = np.median(self.tracks_y)
            if (self.inited == False):  
                self.inited = True
                self.initpos[0] = mx
                self.initpos[1] = my
            else:
                mx = mx - self.initpos[0]
                my = my - self.initpos[1]
                print("error is " + str(my) + " " + str(mx))
               
                self.sky.bump(-mx/40.0, -my/40.0)

        
    def guidesum(self, image):
        if (self.frame_per_guide == 0):
            return
        
        self.idx = self.idx + 1
        sum = sum + frame
        
        if (self.idx == self.frame_per_guide):
            self.idx = 0
            self.curpos = cv2.minMaxLoc(cv2.GaussianBlur(image,(5,5),0))[3]
            mx = self.curpos[0]
            my = self.curpos[1]
            sum = sum * 0
            if (self.inited == False):  
                self.inited = True
                self.initpos[0] = mx
                self.initpos[1] = my
            else:
                mx = mx - self.initpos[0]
                my = my - self.initpos[1]
                print("error is " + str(my) + " " + str(mx))
            
                self.sky.bump(-mx/40.0, -my/40.0)


    def move(self):
        self.initpos[0] += random.uniform(-3,3)
        self.initpos[1] += random.rand(-3,3)
        print("move")

#---------------------------------------------------------------------

def main(args):

    guide_p = guider(args.guide)
    cam_p = emccd(args.gain)

    exp_time_p = args.exp
    print(exp_time_p, args.filename)
    
    cam_p.start(exp_time_p)
    cnt = 1
    tot = 0
    saving = True
    sum = np.zeros((512,512))

    init_ui()

    if (saving):
        base_filename = args.filename + '_' + str(int(time.time())) + '_'
        save_p = saver(base_filename)
        
    while True:
        frame = cam_p.get_frame()
        f1 = frame.astype(float)

        sum = sum + f1
        
        cv2.imshow('live', scale2((1.0/sliders[1]) *  (f1 - sliders[0])))
        cv2.imshow('sum', scale2((1.0/sliders[3]) * (sum/cnt - sliders[2])))
        if (cnt % 10 == 0):
            curpos = cv2.minMaxLoc(cv2.GaussianBlur(f1,(3,3),0))[3]
            if (curpos[0] > 30 and curpos[1] > 30 and curpos[0] < 480 and curpos[1] < 480):
                cv2.imshow('focus', scale3((1.0/sliders[1]) *  (f1[curpos[1]-30:curpos[1]+30, curpos[0]-30:curpos[0]+30] - sliders[0])))

        if (saving):
            save_p.save_data(frame)
        
        guide_p.guide(frame)
               
        if cnt == 100:
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
            guide_p.move()

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
    parser.add_argument("-f", "--filename", type=str, default = '', help="generic file name")
    parser.add_argument("-exp", type=float, default = 0.033, help="exposure in seconds (default 0.033)")
    parser.add_argument("-gain", "--gain", type=int, default = 300, help="emccd gain (default 300)")
    parser.add_argument("-guide", "--guide", type=int, default = 0, help="frame per guide cycle (0 to disable)")
    args = parser.parse_args()
    #t = threading.Thread(target=backgrounder, args=(0,))
    #t.daemon = True
    #t.start()
    print(args)
    time.sleep(10)
    main(args)
     
