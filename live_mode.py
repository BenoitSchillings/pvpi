from pyvcam import pvc
from pyvcam.camera import Camera
from pyvcam import constants as const


import sys
import time
import cv2
import numpy as np
import scipy
import astropy
from astropy.io import fits

sliders = [300, 1000, 300, 1000]


def set(idx, pos):
    sliders[idx] = pos

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
    return(cv2.resize(image, (0,0), fx=1.5, fy=1.5))

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
    pvc.set_param(cam.handle, const.PARAM_GAIN_MULT_FACTOR, 400)
    v = pvc.get_param(cam.handle, const.PARAM_GAIN_MULT_FACTOR, const.ATTR_CURRENT)
    print(cam.temp)



def main(arg):
    init_cam()
    exp_time_p = float(arg[1])
    cam.start_live(exp_time=int(exp_time_p*1000.0))
    print(exp_time_p)
    cnt = 1
    tot = 0
    fps = 0
    cntf = 0
    saving = len(arg) > 2
    sum = np.zeros((512,512))
    k = 0

    init_ui()

    if (saving):
        output = open(arg[2], 'ab')
    while True:
        frame = cam.get_live_frame().reshape(cam.sensor_size[::-1]) 
        f1 = frame.astype(float)

        sum = sum + f1
        f1 = f1 - sliders[0]
        cv2.imshow('live', scale2((1.0/sliders[1]) *  (f1 - sliders[0])))
        s = sum/(cnt)
        cv2.imshow('sum', scale2((1.0/sliders[3]) * (s - sliders[2])))
        cntf = cntf + 1

        if (cnt % 300 == 0):
                print(cnt)
        if (saving and cnt < 9):
                np.save(output, frame)
        if cnt == 1000:
                hdr = fits.header.Header()
                #fits.writeto('single_0.001sec_gain500max_12000' + str(k) + '.fits', np.float32(sum/(cnt*1.0)), hdr, overwrite=True)
                print(cntf)
                sum = np.zeros((512,512))
                k = k + 1
                cnt = 0

        if cv2.waitKey(10) == 27:
            break

        cnt += 1
        tot += 1

    cam.close()
    pvc.uninit_pvcam()
    if (saving):
        output.close()

if __name__ == "__main__":
    main(sys.argv)
