"""Legacy data-ready startup and pose-speed estimate; no RES/AS claims or I/O."""
import math
from .execution_adapter import finite, LocalExecutionAdapter
from .pipeline import Pipeline
from .scene_adapter import SceneAssembler

SOURCE='GPS_pose_difference_filtered_clipped_not_wheel_speed'


class LegacySpeedEstimator(object):
    """Reuse old ds/receive-dt, .7/.3 filtering and clipping semantics.

    Source stamp is retained for age checks. A first sample or long gap requires
    another pose; neither is reported as a measured zero speed.
    """
    def __init__(self,speed_cap):
        if not finite(speed_cap) or speed_cap<=0:raise ValueError('invalid_speed_cap')
        self.cap=speed_cap;self.last=None;self.estimate=0.
        self.unclipped_estimate=0.

    def update(self,pose,stamp,now):
        if len(pose)!=3 or not all(finite(v) for v in pose) or not all(finite(v) for v in (stamp,now)):
            raise ValueError('invalid_speed_pose')
        if not 0<stamp<=now or now-stamp>.8:raise ValueError('speed_pose_stale')
        previous=self.last
        if previous is not None and (stamp<=previous[0] or now<=previous[1]):
            raise ValueError('non_monotonic_speed_pose')
        self.last=(stamp,now,tuple(pose))
        if previous is None or now-previous[1]>.8:
            self.estimate=0.
            self.unclipped_estimate=0.
            raise ValueError('speed_estimate_warming_up')
        dt=now-previous[1]
        if dt<=.001:raise ValueError('speed_estimate_interval_too_short')
        raw_pose_speed=math.hypot(pose[0]-previous[2][0],pose[1]-previous[2][1])/dt
        self.unclipped_estimate=.7*self.unclipped_estimate+.3*raw_pose_speed
        raw=min(raw_pose_speed,self.cap*1.2)
        self.estimate=.7*self.estimate+.3*raw
        return dict(speed=min(max(self.estimate,0.),self.cap),stamp=stamp,
                    source=SOURCE,steering=None,unclipped_speed=self.unclipped_estimate)


class LegacyCompatibleRunner(object):
    """Data-ready candidate runner. Hardware permission is handled externally.

    Task stop is an explicit input, not a fabricated AS state or lap event.
    Full Trackdrive mission completion remains separate from this local runner.
    """
    def __init__(self,vehicle,steering_sign,synthetic=False,max_steering_step=.012,allow_inferred_shadow=False):
        self.planner=Pipeline(vehicle,mode='legacy_compatible')
        self.assembler=SceneAssembler(synthetic=synthetic,mode='legacy_compatible',allow_inferred_shadow=allow_inferred_shadow)
        self.estimator=LegacySpeedEstimator(vehicle['speed_cap'])
        self.execution=LocalExecutionAdapter(steering_sign,vehicle['max_steer'],
                                              max_steering_step=max_steering_step)
        self.fault=None

    def step(self,aligned,free_space,now,version=1,stop_requested=False,emergency=False,motion_authorized=True):
        result=dict(control_mode='legacy_compatible',vehicle_output=False,candidate_valid=False,
                    steering=None,speed_limit=0.,actual_steering=None,actual_speed=None,
                    hardware_go=None,hardware_as_state=None,reason=None,waiting=False,
                    speed_source=SOURCE,external_CAN_monitoring=True)
        try:
            if self.fault:raise ValueError(self.fault)
            if emergency or stop_requested:raise ValueError('external_emergency' if emergency else 'task_stop_requested')
            estimate=self.estimator.update(aligned['pose'],aligned['stamp'],now)
            scene=self.assembler.build(aligned,free_space,estimate,now,version)
            planning=self.planner.step(scene,now,authorized=motion_authorized)
            result['planning']=planning
            if not planning['valid']:raise ValueError(planning['reason'])
            result.update(candidate_valid=True,steering=planning['steering'],
                          speed_limit=planning['speed_limit'],estimated_speed=estimate['speed'],
                          source_stamp=scene.stamp,rear_position=scene.pose[:2])
            result['output_speed_estimate']=estimate['unclipped_speed']
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            reason=str(exc)
            recoverable=reason in ('speed_estimate_warming_up','free_space_unavailable',
                'insufficient_corridor','no_start_gate','pose_speed_estimate_unavailable')
            if not self.execution.started and not self.fault and recoverable:
                result.update(waiting=True,reason=reason)
            else:
                self.fault=reason
                result.update(reason=reason,fault_latched=True)
        execution_candidate=dict(result,candidate_valid=result['candidate_valid'] and motion_authorized)
        result['execution']=self.execution.step(execution_candidate,aligned.get('stamp'),now,
                                                emergency=emergency,stop_requested=stop_requested)
        if self.execution.latched:self.fault=self.execution.latched
        return result
