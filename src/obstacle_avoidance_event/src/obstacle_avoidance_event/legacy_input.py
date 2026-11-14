"""Reuse legacy planar transform semantics without importing a ROS node.

Parameters are explicit legacy-frame values, not a new physical calibration.
Legacy y-right map conversion is a declared basis reflection, not a 3D rotation.
"""
import math
from .execution_adapter import finite
from .model import transform, wrap


class LegacyCoordinates(object):
    def __init__(self, offset_x, offset_y, y_sign, yaw_offset_deg,
                 rear_in_imu=None, convention_confirmed=False):
        values=(offset_x,offset_y,y_sign,yaw_offset_deg)
        if not all(finite(v) for v in values) or y_sign not in (-1,1):
            raise ValueError('invalid_legacy_transform')
        if not convention_confirmed:
            raise ValueError('legacy_y_right_convention_unconfirmed')
        if rear_in_imu is None:
            raise ValueError('imu_to_rear_axle_unavailable')
        if len(rear_in_imu)!=2 or not all(finite(v) for v in rear_in_imu):
            raise ValueError('invalid_rear_offset')
        self.offset_x,self.offset_y=offset_x,offset_y
        self.y_sign,self.yaw_offset_deg=y_sign,yaw_offset_deg
        self.rear=tuple(rear_in_imu) # x-forward/y-left vector, IMU to rear axle.

    def lidar_to_legacy_body(self,x,y):
        if not finite(x) or not finite(y):raise ValueError('invalid_lidar_point')
        lx,ly=x,self.y_sign*y
        if abs(self.yaw_offset_deg)>1e-6:
            angle=math.radians(self.yaw_offset_deg)
            lx,ly=lx*math.cos(angle)-ly*math.sin(angle),lx*math.sin(angle)+ly*math.cos(angle)
        return lx+self.offset_x,ly+self.offset_y

    def lidar_to_rear(self,x,y):
        x,y=self.lidar_to_legacy_body(x,y)
        return (x-self.rear[0],-y-self.rear[1])

    def pose_to_rear(self,state):
        values=[state[k] for k in ('current_x','current_y','current_yaw','origin_yaw')]
        if not all(finite(v) for v in values):raise ValueError('invalid_legacy_pose')
        x,y,current,origin=values
        yaw=wrap(-math.radians(current-origin))
        rear=transform(self.rear,(x,-y,yaw))
        return rear[0],rear[1],yaw
