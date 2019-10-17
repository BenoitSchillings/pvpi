import threading
import time
from inputs import get_gamepad
import controller as xp
import skyx
import copy



bing = 0

def td(inputs):
    global bing
    
    while 1:
        events = get_gamepad()

        for event in events:
            states[event.code] = event.state
            bing = bing + 1
        #print(states)

        
#main

global states
global states0
states = {}
states0 = {}


x = threading.Thread(target=td, args=(0,), daemon=True)
x.start()


pico = True

try:
    xp.init_pico()
except:
    print("no pico")
    pico = False

states["ABS_HAT0Y"] = 0
states["BTN_TL"] = 0
states["ABS_RX"] = 0
states["ABS_RY"] = 0
states["BTN_START"] = 0
states["BTN_SELECT"] = 0

states0 = states.copy()

has_rate = False


sky = skyx.sky6RASCOMTele()
sky.Connect()
pos = sky.GetRaDec()
sky.rate(0, 0)

while 1:
    states0 = copy.copy(states)

    time.sleep(0.032)  

    if (states['BTN_TL'] == 1):
        mult = 50
    else:
        mult = 1
    

    
    #print("main " + str(states['BTN_START']) + " " + str(states0['BTN_START']))
    #print((states['BTN_START']))
    #print((states0['BTN_START']))
    
    if ((states0['BTN_START'] == 1) and (states['BTN_START'] == 0)):
        print("save")
        pos = sky.GetRaDec()
        print(pos)
    
    if (states['BTN_SELECT'] == 0 and states0['BTN_SELECT'] == 1):
        print("gto " + str(pos))
        sky.goto(pos[0], pos[1])
    
    if (states['ABS_HAT0Y'] == 1):
        if (pico):
            xp.move_pico(mult * 100, mult*10)
            print(xp.pos())
            
            
    if (states['ABS_HAT0Y'] == -1):
        if (pico):
            xp.move_pico(mult * 100, mult * -10)
            print(xp.pos())
    
    
    if (abs(states['ABS_RX']) > 500 or abs(states['ABS_RY']) > 500):
       d_dec = states['ABS_RY'] / 1000.0
       d_ra =  states['ABS_RX'] / 1000.0
       d_dec *= mult
       d_ra *= mult
       sky.rate(-d_dec + 0, -d_ra)
       print("move rate is " + str(d_ra) + " " + str(d_dec))
       has_rate = True
    elif (has_rate):
        d_dec = 0
        d_ra = 0
        print("zero")
        has_rate = 0
        sky.rate(d_dec + 0,0) 
    
