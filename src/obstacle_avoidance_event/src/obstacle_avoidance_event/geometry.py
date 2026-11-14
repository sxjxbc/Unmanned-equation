"""Conservative union of possible overhang splits, rear-axle reference."""
import math

def body_disks(point, yaw, vehicle, extra=0.):
    if vehicle.get('geometry_mode') not in ('unknown_overhang_union','imu_body_rectangle','rear_axle_rectangle'):
        yield point,math.hypot(max(vehicle['front'],vehicle['rear']),vehicle['width']/2.)+vehicle['margin']+extra
        return
    # Given only length and wheelbase: rear overhang in [0,length-wheelbase].
    # Union rectangle is [-rear_bound,+front_bound] x [-width/2,+width/2].
    rear,front=vehicle['rear'],vehicle['front']
    n=max(1,int(math.ceil((front+rear)/.25)))
    step=(front+rear)/n
    radius=math.hypot(step/2.,vehicle['width']/2.)+vehicle['margin']+extra
    for i in range(n):
        x=-rear+(i+.5)*step
        yield (point[0]+math.cos(yaw)*x,point[1]+math.sin(yaw)*x),radius

def design_vehicle(profile, assumptions):
    if profile['localization_offset'] is not None:
        raise ValueError('profile_contract_changed')
    v=dict(assumptions)
    v.update(wheelbase=profile['wheelbase'],width=profile['width'],
             front=profile['length'],rear=profile['length']-profile['wheelbase'],
             geometry_mode='unknown_overhang_union')
    return v


def imu_body_vehicle(profile, margin):
    """User-authorized assumed body footprint. NOT a rear-axle controller profile."""
    body=profile['imu_body_assumption']
    if margin<0:raise ValueError('negative_margin')
    return dict(front=body['front_extent_m'],rear=body['rear_extent_m'],
                width=2*body['lateral_envelope_half_width_m'],margin=margin,geometry_mode='imu_body_rectangle',
                pose_reference='imu',wheelbase=profile['wheelbase'])

def lidar_point_in_imu(point, profile, rotation):
    """3D proper rotation must be supplied; translation alone cannot define axes."""
    if rotation is None:raise ValueError('lidar_rotation_unconfirmed')
    if len(rotation)!=3 or any(len(row)!=3 for row in rotation):raise ValueError('invalid_rotation_shape')
    if any(math.isnan(v) or math.isinf(v) for row in rotation for v in row):raise ValueError('invalid_rotation')
    for i in range(3):
        for j in range(3):
            dot=sum(rotation[i][k]*rotation[j][k] for k in range(3))
            if abs(dot-(1. if i==j else 0.))>1e-6:raise ValueError('non_orthonormal_rotation')
    a,b,c=rotation
    det=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(det-1.)>1e-6:raise ValueError('reflection_is_not_rotation')
    t=profile['lidar_translation_in_imu_m']
    return tuple(sum(rotation[i][j]*point[j] for j in range(3))+t[i] for i in range(3))

def imu_pose_to_rear_axle(pose, profile):
    offset=profile['imu_to_rear_axle_m']
    if offset is None:raise ValueError('imu_to_rear_axle_unavailable')
    x,y,yaw=pose
    return (x+math.cos(yaw)*offset[0]-math.sin(yaw)*offset[1],
            y+math.sin(yaw)*offset[0]+math.cos(yaw)*offset[1],yaw)


def imu_point_in_map(point, imu_pose):
    x,y,yaw=imu_pose
    return (x+math.cos(yaw)*point[0]-math.sin(yaw)*point[1],
            y+math.sin(yaw)*point[0]+math.cos(yaw)*point[1],point[2])
