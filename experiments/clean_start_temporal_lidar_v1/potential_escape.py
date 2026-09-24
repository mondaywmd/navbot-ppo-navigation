"""Full-scan repulsive-gradient escape controller with a hard safety envelope."""
import math
import numpy as np

from safety_shield import filter_command, valid_scan

INFLUENCE_DISTANCE = 0.90
COLLISION_DISTANCE = 0.20
HARD_SAFETY_DISTANCE = 0.40
NO_FORWARD_DISTANCE = 0.40
ACTIVATE_DISTANCE = 0.60
RELEASE_DISTANCE = 0.75
MIN_GOAL_CLEARANCE = 0.25
GOAL_RELAX_RADIUS = 0.50
SWEEP_HALF_WIDTH = 0.22


def desired_clearance(goal_distance):
    """One metre normally, smoothly relaxed only inside the goal neighbourhood."""
    scale=float(np.clip(goal_distance/GOAL_RELAX_RADIUS,0.0,1.0))
    return MIN_GOAL_CLEARANCE+(HARD_SAFETY_DISTANCE-MIN_GOAL_CLEARANCE)*scale


def wrap(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def obstacle_gradient(scan):
    """Return normalized 2-D direction away from all relevant scan points."""
    values = valid_scan(scan)
    angles = scan.angle_min + np.arange(len(values)) * scan.angle_increment
    mask = values < INFLUENCE_DISTANCE
    if not np.any(mask):
        return np.array([0.0, 0.0]), float(values.min()), float("inf"), 0.0
    d = values[mask]
    a = angles[mask]
    # Smooth compact-support potential: nearby surfaces dominate without a
    # discontinuity when a ray enters/leaves the influence radius.
    weights = ((INFLUENCE_DISTANCE - d) / INFLUENCE_DISTANCE) ** 2 / np.maximum(d, 0.12) ** 2
    away = np.stack((-np.cos(a), -np.sin(a)), axis=1)
    vector = (weights[:, None] * away).sum(axis=0)
    norm = float(np.linalg.norm(vector))
    strength = norm / len(values)
    if norm > 1e-9:
        vector /= norm
    # Robust clearance trend: mean of the five closest rays, not one noisy ray.
    closest = np.partition(values, min(4, len(values)-1))[:min(5, len(values))]
    return vector, float(values.min()), float(closest.mean()), strength


def project_goal_to_safe_direction(goal, gradient, required_outward):
    """Minimum-change projection satisfying dot(direction, gradient)>=required."""
    goal=np.asarray(goal,dtype=float);gradient=np.asarray(gradient,dtype=float)
    correction=max(0.0,float(required_outward-np.dot(goal,gradient)))
    direction=goal+correction*gradient
    norm=float(np.linalg.norm(direction))
    return direction/norm if norm>1e-9 else gradient.copy()


def constrained_safe_direction(scan, goal_heading, required_clearance, activation_distance):
    """Choose a direction satisfying every nearby LiDAR radial constraint."""
    values=valid_scan(scan)
    angles=scan.angle_min+np.arange(len(values))*scan.angle_increment
    mask=values<activation_distance
    goal=np.array([math.cos(goal_heading),math.sin(goal_heading)])
    if not np.any(mask):return goal,True,float("inf")
    d=values[mask];a=angles[mask]
    candidates=np.linspace(-math.pi,math.pi,144,endpoint=False)
    # Positive required rate inside the safety boundary; a small amount of
    # approach is tolerated only in the anticipatory band outside it.
    required_rate=np.clip(0.40*(required_clearance-d)/required_clearance,-0.08,0.45)
    relative=a[:,None]-candidates[None,:]
    radial=-np.cos(relative)
    along=d[:,None]*np.cos(relative)
    lateral=np.abs(d[:,None]*np.sin(relative))
    # A side obstacle outside the swept body corridor does not constrain
    # straight translation.  It matters only for candidate directions whose
    # future corridor actually intersects it.
    in_swept_corridor=(along>0.0)&(lateral<=SWEEP_HALF_WIDTH)
    margins=np.where(in_swept_corridor,radial-required_rate[:,None],np.inf).min(axis=0)
    feasible=margins>=0.0
    goal_progress=np.cos(candidates-goal_heading)
    if np.any(feasible):
        scores=np.where(feasible,goal_progress+0.05*np.minimum(margins,1.0),-np.inf)
        index=int(np.argmax(scores));is_feasible=True
    else:
        # When overlapping safety regions make instantaneous satisfaction
        # impossible, maximize the worst constraint first, goal progress second.
        best=float(margins.max());near_best=margins>=best-1e-6
        index=int(np.argmax(np.where(near_best,goal_progress,-np.inf)));is_feasible=False
    direction=np.array([math.cos(candidates[index]),math.sin(candidates[index])])
    return direction,is_feasible,float(margins[index])


class PotentialEscapeShield:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.previous_clearance = None
        self.growth_streak = 0

    def apply(self, linear, angular, scan, front_ttc=float("inf"), turn_side_ttc=float("inf"),
              goal_distance=float("inf")):
        gradient, nearest, clearance, strength = obstacle_gradient(scan)
        required_clearance=desired_clearance(goal_distance)
        activate_distance=required_clearance+0.20
        release_distance=required_clearance+0.35
        if nearest <= COLLISION_DISTANCE:
            return 0.0, 0.0, "potential_collision_stop"
        if not self.active and nearest <= activate_distance:
            self.active=True;self.previous_clearance=clearance;self.growth_streak=0
        if self.active:
            growing = self.previous_clearance is not None and clearance - self.previous_clearance >= 0.003
            self.growth_streak = self.growth_streak + 1 if growing else 0
            self.previous_clearance = clearance
            if nearest >= release_distance and self.growth_streak >= 3:
                self.reset()
                out=filter_command(linear,angular,scan,front_ttc,turn_side_ttc)
                return out[0],out[1],"potential_release_"+out[2]
            # Recover the goal-heading error used by the waypoint controller.
            goal_heading = float(np.clip(angular / 1.8, -math.pi / 2, math.pi / 2))
            goal = np.array([math.cos(goal_heading), math.sin(goal_heading)])
            danger=float(np.clip((activate_distance-nearest)/
                                 (activate_distance-COLLISION_DISTANCE),0.0,1.0))
            # Preserve as much goal progress as possible.  The repulsive
            # gradient contributes only the minimum correction needed to meet
            # an outward-motion constraint that tightens near an obstacle.
            direction,all_constraints_met,worst_margin=constrained_safe_direction(
                scan,goal_heading,required_clearance,activate_distance)
            heading = math.atan2(float(direction[1]), float(direction[0]))
            drive_sign = 1.0
            alignment_heading = heading
            if abs(heading) > math.pi/2:
                drive_sign = -1.0
                alignment_heading = wrap(heading-math.copysign(math.pi,heading))
            turn = float(np.clip(1.6 * alignment_heading, -0.8, 0.8))

            # Linear motion is allowed only when already aimed near the vector.
            forward = 0.0
            if abs(alignment_heading) <= math.radians(35):
                desired = min(0.18, max(0.06, float(linear)))
                values = valid_scan(scan)
                angles = scan.angle_min + np.arange(len(values)) * scan.angle_increment
                # Positive means the chosen translation increases clearance
                # from every surface in its actual swept corridor.
                forward_along=drive_sign*values*np.cos(angles)
                forward_lateral=np.abs(values*np.sin(angles))
                threat_mask=(forward_along>0.0)&(forward_lateral<=SWEEP_HALF_WIDTH)&(values<activate_distance)
                outward_rate=(float(np.min(-drive_sign*np.cos(angles[threat_mask])))
                              if np.any(threat_mask) else float("inf"))
                corridor_nearest=(float(values[threat_mask].min()) if np.any(threat_mask)
                                  else float("inf"))
                if not np.any(threat_mask):
                    forward=drive_sign*0.18
                elif corridor_nearest > required_clearance:
                    scale = float(np.clip((corridor_nearest-required_clearance)/0.18, 0.20, 1.0))
                    forward = drive_sign*desired*scale
                elif outward_rate > 0.10:
                    forward = drive_sign*(min(0.08,desired) if drive_sign<0 else min(0.04,desired))
                # Inside the doubled hard-safety band, translation is legal
                # only when all closest surfaces have positive clearance rate.
                if corridor_nearest <= required_clearance and outward_rate <= 0.10:
                    forward = 0.0
                if self.growth_streak >= 2:
                    forward = float(np.clip(forward*1.35,-0.10,0.18))
            prefix="barrier" if all_constraints_met else "barrier_recovery"
            reason = prefix+"_reverse" if forward<0 else (prefix+"_depart" if forward>0 else prefix+"_align")
            return forward, turn, reason
        return filter_command(linear, angular, scan, front_ttc, turn_side_ttc)
