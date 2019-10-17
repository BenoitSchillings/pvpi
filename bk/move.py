import time
from inputs import get_gamepad
import controller as xp
import sys


if (len(sys.argv) == 2):
    v1 = sys.argv[1]
    v2 = v1
    v3 = v1
else:
    v1 = sys.argv[1] 
    v2 = sys.argv[2]
    v3 = sys.argv[3]

v1 = int(v1)
v2 = int(v2)
v3 = int(v3)

print(str(v1) + " " + str(v2) + " " + str(v3))

xp.init_pico()
xp.move3(500, v1, v2, v3)

    
