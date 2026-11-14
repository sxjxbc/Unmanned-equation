"""Output request guard. Caller supplies the original protocol client.

No speed/throttle command exists. A software cap is NOT physical speed control.
"""
import math
from .execution_adapter import finite


class LowSpeedOutputGuard(object):
    def __init__(self,client=None,enabled=False,max_age=.5,max_steer=.5,
                 max_step=.012,max_speed=1.,distance_limit=3.,duration_limit=15.):
        if (not all(finite(v) and v>0 for v in (max_age,max_steer,max_step,max_speed))
                or not all(finite(v) and v>=0 for v in (distance_limit,duration_limit))):
            raise ValueError('invalid_output_limits')
        if enabled and client is None:raise ValueError('output_client_required')
        self.client,self.enabled=client,enabled
        self.age,self.max_steer,self.step_limit=max_age,max_steer,max_step
        self.max_speed,self.distance_limit,self.duration_limit=max_speed,distance_limit,duration_limit
        self.latest=None;self.last_received=None;self.start=None;self.progress=0.
        self.last_position=None;self.position_stamp=None;self.last_now=None
        self.previous_steer=0.;self.fault=None

    def receive(self,packet,now):
        if not isinstance(packet,dict):self.fault=self.fault or 'malformed_candidate';return
        self.latest=dict(packet);self.last_received=now

    def tick(self,now,channel_conflict=False):
        out=dict(action='wait',enabled=self.enabled,stop_reason=self.fault,
                 distance_m=self.progress,steering=None,send_ok=None,
                 physical_speed_control=False,distance_limit_m=self.distance_limit,
                 duration_limit_s=self.duration_limit)
        if not self.enabled:return out
        try:
            if self.fault:raise ValueError(self.fault)
            if not finite(now) or (self.last_now is not None and now<self.last_now):raise ValueError('output_clock_error')
            self.last_now=now
            if channel_conflict:raise ValueError('output_channel_conflict')
            if self.duration_limit>0 and self.start is not None and now-self.start>=self.duration_limit:raise ValueError('test_duration_complete')
            p=self.latest
            if p is None:
                if self.start is not None:raise ValueError('candidate_missing')
                return out
            if p.get('stop_requested') is True:raise ValueError('candidate_stop_requested')
            if p.get('valid') is not True or p.get('drive_eligible') is not True:
                if self.start is not None:raise ValueError('candidate_not_drive_eligible')
                return out
            stamp=p.get('source_stamp')
            if not finite(stamp) or not 0<=now-stamp<=self.age or not finite(self.last_received) or not 0<=now-self.last_received<=self.age:
                raise ValueError('output_candidate_stale')
            angle,cap,speed=[p.get(k) for k in ('steering','speed_limit','estimated_speed')]
            if not all(finite(v) for v in (angle,cap,speed)) or abs(angle)>self.max_steer or cap<=0 or speed<0:
                raise ValueError('invalid_output_candidate')
            if speed>min(cap,self.max_speed)+.05:raise ValueError('estimated_speed_above_test_limit')
            position=p.get('rear_position')
            if not isinstance(position,(tuple,list)) or len(position)!=2 or not all(finite(v) for v in position):
                raise ValueError('rear_position_required')
            if self.position_stamp is not None and stamp<self.position_stamp:raise ValueError('output_progress_time_rewind')
            if self.position_stamp is not None and stamp==self.position_stamp and tuple(position)!=self.last_position:
                raise ValueError('position_changed_without_new_stamp')
            if self.position_stamp is None or stamp>self.position_stamp:
                if self.last_position is not None:
                    ds=math.hypot(position[0]-self.last_position[0],position[1]-self.last_position[1])
                    if ds>self.max_speed*(stamp-self.position_stamp)+.4:raise ValueError('test_position_jump')
                    self.progress+=ds
                self.last_position=tuple(position);self.position_stamp=stamp
            if self.distance_limit>0 and self.progress>=self.distance_limit:raise ValueError('test_distance_complete')
            angle=max(self.previous_steer-self.step_limit,min(self.previous_steer+self.step_limit,angle))
            if not self.client.send_steering_frame(angle,stop=False):raise ValueError('steering_send_failed')
            self.previous_steer=angle
            if self.start is None:self.start=now
            out.update(action='steer',steering=angle,send_ok=True,distance_m=self.progress)
            return out
        except Exception as exc:
            self.fault=self.fault or str(exc)
        try:ok=bool(self.client.send_steering_frame(0.,stop=True))
        except Exception:ok=False
        out.update(action='stop',stop_reason=self.fault,send_ok=ok,distance_m=self.progress)
        return out

    def shutdown(self):
        if not self.enabled:return
        try:self.client.send_steering_frame(0.,stop=True)
        finally:self.client.close()
