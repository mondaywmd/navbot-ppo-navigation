"""Stateful LiDAR-only tangential escape used when stop constraints interlock."""
import math
import numpy as np
from safety_shield import HARD_DISTANCE, SLOW_DISTANCE, filter_command, valid_scan

ACTIVATE_DISTANCE=0.40; RELEASE_DISTANCE=0.42; COLLISION_DISTANCE=0.20
NO_FORWARD_DISTANCE=0.30; FULL_TANGENT_SPEED_DISTANCE=0.45
# Pure 90-degree tangency preserves clearance.  Keeping the obstacle slightly
# behind the side plane adds an outward radial component while still avoiding
# the inefficient "turn completely away" behaviour.
TARGET_SIDE_BEARING=math.radians(105); ALIGN_TOLERANCE=math.radians(25)
RELEASE_SIDE_BEARING=math.radians(65); RELEASE_GROWTH=0.003
RELEASE_STREAK=3; COOLDOWN_STEPS=10
OUTWARD_BEARING=math.radians(95); OUTWARD_MIN_SCALE=0.25

def wrap(value):return (value+math.pi)%(2*math.pi)-math.pi

def nearest_bearing(scan):
    values=valid_scan(scan);angles=scan.angle_min+np.arange(len(values))*scan.angle_increment
    wrapped=(angles+math.pi)%(2*math.pi)-math.pi
    relevant=np.abs(wrapped)<=math.radians(120);indices=np.where(relevant)[0]
    index=indices[int(np.argmin(values[indices]))]
    return float(wrapped[index]),float(values[index])

def side_clearances(scan):
    values=valid_scan(scan);angles=scan.angle_min+np.arange(len(values))*scan.angle_increment
    wrapped=(angles+math.pi)%(2*math.pi)-math.pi
    left=float(values[(wrapped>0)&(wrapped<=math.pi/2)].min())
    right=float(values[(wrapped<0)&(wrapped>=-math.pi/2)].min())
    return left,right

class TangentEscapeShield:
    def __init__(self):self.reset();self.cooldown=0
    def reset(self):
        self.active=False;self.pass_sign=0;self.previous_distance=None;self.growth_streak=0
    def apply(self,linear,angular,scan,front_ttc=float("inf"),turn_side_ttc=float("inf")):
        bearing,distance=nearest_bearing(scan)
        if self.cooldown>0:self.cooldown-=1
        if self.active:
            growing=self.previous_distance is not None and distance-self.previous_distance>=RELEASE_GROWTH
            self.growth_streak=self.growth_streak+1 if growing else 0
            self.previous_distance=distance
            safely_beside=abs(bearing)>=RELEASE_SIDE_BEARING
            if distance>=RELEASE_DISTANCE and safely_beside and self.growth_streak>=RELEASE_STREAK:
                self.reset();self.cooldown=COOLDOWN_STEPS
                out=filter_command(linear,angular,scan,front_ttc,turn_side_ttc)
                return out[0],out[1],"escape_release_"+out[2]
        if not self.active and self.cooldown==0 and distance<=ACTIVATE_DISTANCE:
            left,right=side_clearances(scan)
            self.pass_sign=(1 if angular>0.05 else (-1 if angular<-0.05 else (1 if left>=right else -1)))
            self.active=True;self.previous_distance=distance;self.growth_streak=0
        if self.active:
            desired_bearing=-self.pass_sign*TARGET_SIDE_BEARING
            error=wrap(bearing-desired_bearing)
            turn=float(np.clip(1.2*error,-0.8,0.8))
            # Escape chooses a direction, but never bypasses the independent
            # nearest-clearance envelope.  A differential-drive robot first
            # turns in place when too close, then gains tangential speed as
            # clearance grows.
            if distance<=COLLISION_DISTANCE:return 0.0,0.0,"collision_stop"
            desired_forward=0.03 if abs(error)>ALIGN_TOLERANCE else 0.15
            clearance_scale=0.0 if distance<=NO_FORWARD_DISTANCE+1e-6 else float(np.clip(
                (distance-NO_FORWARD_DISTANCE)/(FULL_TANGENT_SPEED_DISTANCE-NO_FORWARD_DISTANCE),0.0,1.0))
            # Once the closest obstacle is behind the side plane, positive
            # forward motion has a strictly outward radial component
            # (-v*cos(bearing) > 0).  Permit a small escape velocity there so
            # an in-place turn cannot deadlock below NO_FORWARD_DISTANCE.
            if abs(bearing)>=OUTWARD_BEARING:
                clearance_scale=max(clearance_scale,OUTWARD_MIN_SCALE)
            forward=desired_forward*clearance_scale
            if forward==0.0:
                reason="tangent_rotate_only"
            else:
                reason="tangent_turn" if abs(error)>ALIGN_TOLERANCE else "tangent_depart"
            return forward,turn,reason
        return filter_command(linear,angular,scan,front_ttc,turn_side_ttc)
