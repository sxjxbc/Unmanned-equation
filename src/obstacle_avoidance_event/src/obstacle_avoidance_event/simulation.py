"""Synthetic kinematic bicycle. Not a real actuator or calibrated model."""
import math
from .model import wrap

class Bicycle(object):
    def __init__(self, vehicle, pose=(.5,0.,0.)):
        self.vehicle=vehicle
        self.pose=pose
        self.speed=0.
        self.steering=0.

    def step(self, steering, speed, dt):
        if dt <= 0 or not all(not math.isnan(v) and not math.isinf(v) for v in (dt,steering,speed)):
            raise ValueError('invalid_simulation_input')
        v=self.vehicle
        target=max(-v['max_steer'],min(v['max_steer'],steering))
        self.steering+=max(-v['steer_rate']*dt,min(v['steer_rate']*dt,target-self.steering))
        old=self.speed
        rate=v['brake'] if speed<old else v.get('acceleration',.3)
        self.speed=max(0.,old+max(-rate*dt,min(rate*dt,speed-old)))
        mean=(old+self.speed)/2.
        x,y,yaw=self.pose
        omega=mean*math.tan(self.steering)/v['wheelbase']
        # Midpoint integration; slip and actuator latency are not modeled.
        self.pose=(x+mean*math.cos(yaw+omega*dt/2.)*dt,
                   y+mean*math.sin(yaw+omega*dt/2.)*dt,wrap(yaw+omega*dt))
        return self.pose


def audit_motion(before, after, scene, vehicle, travel):
    """Whole body circumscribed disk plus traveled distance bounds this step.

    Conservative for any yaw during the step; requires free() disk semantics.
    """
    from .model import distance
    radius=math.hypot(max(vehicle['front'],vehicle['rear']),vehicle['width']/2.)+vehicle['margin']
    bound=radius+travel
    if scene.free is None or not scene.free(before[:2],bound):
        return 'actual_motion_unknown_or_boundary'
    for x,y,r in scene.obstacles+[(x,y,.15) for x,y in scene.cones]:
        if distance(before[:2],(x,y)) <= bound+r:
            return 'actual_motion_collision'
    return None
