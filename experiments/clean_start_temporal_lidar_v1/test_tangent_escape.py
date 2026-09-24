import math
import numpy as np
from tangent_escape import TangentEscapeShield

class Scan:
    angle_min=-math.pi;angle_increment=2*math.pi/360;range_min=0.12;range_max=3.5
    def __init__(self,index,distance):self.ranges=np.full(360,3.5);self.ranges[index]=distance

left=TangentEscapeShield();command=left.apply(0.2,0.4,Scan(180,0.38))
assert left.active and left.pass_sign==1 and command[1]>0 and command[2]=="tangent_turn"
aligned=left.apply(0.2,0.4,Scan(75,0.35))
assert 0.04 < aligned[0] < 0.06 and abs(aligned[1])<1e-6 and aligned[2]=="tangent_depart"
too_close=left.apply(0.2,0.4,Scan(90,0.25))
assert too_close[0]==0.0 and too_close[2]=="tangent_rotate_only"
outward_close=left.apply(0.2,0.4,Scan(75,0.25))
assert outward_close[0]>0.0 and outward_close[2]=="tangent_depart"
scaled=left.apply(0.2,0.4,Scan(90,0.30))
assert scaled[0]==0.0
full=left.apply(0.2,0.4,Scan(75,0.45))
assert math.isclose(full[0],0.15,rel_tol=1e-5,abs_tol=1e-6)
# Release only after clearance grows for three consecutive frames while the
# obstacle is safely beside the robot; one isolated large reading is not enough.
released=left.apply(0.2,0.4,Scan(75,0.455))
assert not left.active and left.cooldown==10 and released[2].startswith("escape_release_")
# Cooldown prevents immediate re-entry on a still-near scan.
held=left.apply(0.2,0.4,Scan(90,0.39))
assert not left.active and left.cooldown==9 and not held[2].startswith("tangent_")
right=TangentEscapeShield();command=right.apply(0.2,-0.4,Scan(180,0.38))
assert right.pass_sign==-1 and command[1]<0
print("TANGENT_ESCAPE_TEST_PASS")
