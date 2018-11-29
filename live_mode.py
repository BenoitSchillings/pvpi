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

sliders = [300, 1000, 300, 1000]

#---------------------------------------------------------------------

def set(idx, pos):
    sliders[idx] = pos

#---------------------------------------------------------------------

def save_fits(nparray):
    hdr = fits.header.Header()
    fits.writeto('fn.fits', np.float32(nparray), hdr, overwrite=True)

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

def scale2(image):
    return(cv2.resize(image, (0,0), fx=1.0, fy=1.0))

#---------------------------------------------------------------------

def init_cam():
    pvc.init_pvcam()
    global cam

    cam = next(Camera.detect_camera())
    cam.open()
    cam.gain=3
    print(cam.temp)
    cam.temp_setpoint = -8000
    print(cam.temp_setpoint)

    pvc.set_param(cam.handle, const.PARAM_READOUT_PORT, 0)
    pvc.set_param(cam.handle, const.PARAM_GAIN_MULT_FACTOR, 40)
    v = pvc.get_param(cam.handle, const.PARAM_GAIN_MULT_FACTOR, const.ATTR_CURRENT)
    print(cam.temp)

#---------------------------------------------------------------------

def save_data(data, filename, index, frame):
    global output_file
    if (frame == 1):
        output_file = open(filename + str(index), 'ab')
    
    np.save(output_file, data)
    if (frame == FRAME_PER_FILE):
        output_file.close()
        
#---------------------------------------------------------------------
 
def close_files():
    if (output_file != None):
        output_file.close()
        
#---------------------------------------------------------------------


def main(arg):
    init_cam()
    exp_time_p = float(arg[1])
    cam.start_live(exp_time=int(exp_time_p*1000.0))
    cnt = 1
    tot = 0
    saving = len(arg) > 2
    sum = np.zeros((512,512))
    seq = 0

    init_ui()

    if (saving):
        base_filename = arg[2] + '_' + str(int(time.time())) + '_'
    
    while True:
        frame = cam.get_live_frame().reshape(cam.sensor_size[::-1]) 
        f1 = frame.astype(float)

        sum = sum + f1
        
        cv2.imshow('live', scale2((1.0/sliders[1]) *  (f1 - sliders[0])))
        cv2.imshow('sum', scale2((1.0/sliders[3]) * (sum/cnt - sliders[2])))
        
        if (saving):
            save_data(frame, base_filename, seq, cnt)
                
        if cnt == FRAME_PER_FILE:
            print("file # " + str(seq) + " total frame = " + str(tot))
            sum = np.zeros((512,512))
            seq = seq + 1
            cnt = 0

        if cv2.waitKey(10) == 27:
            break

        cnt += 1
        tot += 1

    cam.close()
    pvc.uninit_pvcam()
    close_files()
    
#---------------------------------------------------------------------

if __name__ == "__main__":
    main(sys.argv)
