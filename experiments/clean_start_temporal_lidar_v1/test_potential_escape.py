import math
import numpy as np
from potential_escape import (PotentialEscapeShield, desired_clearance,
                              constrained_safe_direction, obstacle_gradient,
                              project_goal_to_safe_direction)

class Scan:
    angle_min=-math.pi;angle_increment=2*math.pi/360;range_min=0.12;range_max=3.5
    def __init__(self, points=()):
        self.ranges=np.full(360,3.5)
        for bearing,distance in points:
            self.ranges[int(round((bearing-self.angle_min)/self.angle_increment))%360]=distance

def cluster(bearing,distance):
    return [(bearing+math.radians(k),distance) for k in (-2,-1,0,1,2)]

# A left obstacle produces a rightward component; mirroring reverses only y.
g,d,_,s1=obstacle_gradient(Scan([(math.radians(35),0.35),(math.radians(45),0.36)]))
gm,dm,_,s2=obstacle_gradient(Scan([(math.radians(-35),0.35),(math.radians(-45),0.36)]))
assert g[0] < 0 and g[1] < 0 and np.allclose(gm,[g[0],-g[1]],atol=1e-12) and d==dm
assert math.isclose(s1,s2,rel_tol=1e-12)
assert desired_clearance(2.0)==0.40
assert 0.25 < desired_clearance(0.25) < 0.40

# Symmetric obstacles cancel lateral force and combine longitudinal repulsion.
g,_,_,_=obstacle_gradient(Scan([(math.radians(30),0.35),(math.radians(-30),0.35)]))
assert g[0] < -0.999 and abs(g[1]) < 1e-12

# Projection changes the goal by the minimum gradient component required for
# outward motion, rather than letting repulsion become the navigation target.
goal=np.array([1.0,0.0]);gradient=np.array([0.0,-1.0])
safe=project_goal_to_safe_direction(goal,gradient,0.5)
assert safe[0]>0 and safe[1]<0 and np.dot(safe,gradient)>=0.44

# Every active sector is constrained independently: two front-side obstacles
# select a backwards escape, not their potentially cancelling average.
scan=Scan(cluster(math.radians(35),0.32)+cluster(math.radians(-35),0.32))
safe,feasible,margin=constrained_safe_direction(scan,0.0,0.40,0.60)
assert feasible and safe[0]<0.20 and abs(safe[1])>0.90 and margin>=0

s=PotentialEscapeShield()
reverse=s.apply(0.2,0.0,Scan(cluster(0.0,0.32)))
assert reverse[0] <= 0.0 and (reverse[2].endswith("_reverse") or reverse[2].endswith("_align"))
# A pure side obstacle outside the swept corridor must not stop straight motion.
s=PotentialEscapeShield()
turn=s.apply(0.2,0.4,Scan(cluster(math.radians(90),0.32)))
assert turn[0]>=0.17 and turn[2].endswith("_depart")
# If the closest surface is behind, a small forward command strictly escapes.
s=PotentialEscapeShield()
away=s.apply(0.2,0.0,Scan([(math.radians(170),0.39)]))
assert away[0]>=0.17 and away[2].endswith("_depart")
# Collision floor remains unconditional.
stop=s.apply(0.2,0.0,Scan([(math.radians(120),0.19)]))
assert stop==(0.0,0.0,"potential_collision_stop")
# The raw field itself fades continuously with distance.
_,_,_,near_strength=obstacle_gradient(Scan([(0.0,0.35)]))
_,_,_,far_strength=obstacle_gradient(Scan([(0.0,1.40)]))
assert near_strength > far_strength == 0
# Escape ends only with both safe clearance and a repeated increasing trend.
s=PotentialEscapeShield();s.apply(0.2,0.0,Scan(cluster(0.0,0.35)))
s.apply(0.2,0.0,Scan(cluster(math.radians(170),0.76)))
s.apply(0.2,0.0,Scan(cluster(math.radians(170),0.77)))
released=s.apply(0.2,0.0,Scan(cluster(math.radians(170),0.78)))
assert not s.active and released[2].startswith("potential_release_")
print("POTENTIAL_ESCAPE_TEST_PASS")
