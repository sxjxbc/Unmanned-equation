"""Explicit internal contracts, not definitions of existing hardware messages."""
from .model import Scene
from .execution_adapter import finite


class SceneAssembler(object):
    def __init__(self, synthetic=False, mode='strict',allow_inferred_shadow=False):
        if mode not in ('strict','legacy_compatible'):raise ValueError('invalid_control_mode')
        self.mode=mode
        if allow_inferred_shadow and mode!='legacy_compatible':raise ValueError('inferred_boundary_requires_shadow_compatibility')
        self.allow_inferred_shadow=allow_inferred_shadow
        self.synthetic=synthetic
        self.last_stamp=None
        self.version=None

    def build(self, aligned, free_space, feedback, now, version):
        stamp=aligned['stamp']
        if not finite(now) or not finite(stamp) or not 0 < stamp <= now or now-stamp>.5:
            raise ValueError('aligned_observation_stale')
        if aligned.get('pose_reference')!='rear_axle':
            raise ValueError('rear_axle_reference_unconfirmed')
        frame=aligned.get('frame')
        if not frame:raise ValueError('map_frame_unconfirmed')
        if self.last_stamp is not None and stamp<=self.last_stamp:
            raise ValueError('non_monotonic_scene')
        if self.version is not None and version!=self.version:
            raise ValueError('map_version_changed')
        if not free_space or not callable(free_space.get('free')):
            raise ValueError('free_space_unavailable')
        if free_space.get('frame')!=frame or free_space.get('version')!=version:
            raise ValueError('free_space_frame_or_version_mismatch')
        ft=free_space.get('stamp')
        if not finite(ft) or abs(ft-stamp)>.001:
            raise ValueError('free_space_time_mismatch')
        sources=('synthetic',) if self.synthetic else ('confirmed_boundary','certified_sensor')
        if self.allow_inferred_shadow:sources+=('inferred_boundary_shadow_only','inferred_boundary_low_speed_trial')
        if free_space.get('source') not in sources:
            raise ValueError('free_space_source_unconfirmed')
        if not feedback:raise ValueError('feedback_unavailable')
        if self.mode=='legacy_compatible':
            if feedback.get('source')!='GPS_pose_difference_filtered_clipped_not_wheel_speed':
                raise ValueError('pose_speed_estimate_unavailable')
            speed,ts=[feedback.get(k) for k in ('speed','stamp')]
            if feedback.get('steering') is not None:
                raise ValueError('estimated_steering_is_not_feedback')
            if not all(finite(v) for v in (speed,ts)) or speed<0 or not 0<=now-ts<=.8:
                raise ValueError('invalid_or_stale_speed_estimate')
            steer=None
        else:
            if feedback.get('source')!=('synthetic' if self.synthetic else 'hardware'):
                raise ValueError('feedback_source_unconfirmed')
            speed,steer,ts=[feedback.get(k) for k in ('speed','steering','stamp')]
            if not all(finite(v) for v in (speed,steer,ts)) or speed<0 or not 0<=now-ts<=.2:
                raise ValueError('invalid_or_stale_feedback')
        pose,cones=aligned['pose'],aligned['cones']
        if len(pose)!=3 or not all(finite(v) for v in pose):raise ValueError('invalid_aligned_pose')
        if len(cones)>120 or any(len(p)!=2 or not all(finite(v) for v in p) for p in cones):
            raise ValueError('invalid_aligned_cones')
        obstacles=aligned.get('obstacles',[])
        if any(len(p)!=3 or not all(finite(v) for v in p) or p[2]<0 for p in obstacles):
            raise ValueError('invalid_aligned_obstacles')
        scene=Scene(cones,pose,stamp,version=version,free=free_space['free'],
                    feedback=(speed,steer,ts) if self.mode=='strict' else None,
                    speed_estimate=dict(feedback) if self.mode=='legacy_compatible' else None,
                    obstacles=obstacles)
        self.last_stamp,self.version=stamp,version
        return scene
