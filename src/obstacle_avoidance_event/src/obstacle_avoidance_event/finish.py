"""Candidate-only startup and finish adapters; no official AS state commands."""
import math

def value(signals,key,now,budget=.25):
    if key not in signals:raise ValueError('missing_'+key)
    v,stamp=signals[key]
    if not 0<=now-stamp<=budget:raise ValueError('stale_'+key)
    return v

class StartupGate(object):
    def __init__(self):
        self.ready_since=None;self.last_go=False;self.go_accepted=False;self.fault=None
    def step(self,now,signals):
        try:
            if self.fault:raise ValueError(self.fault)
            if value(signals,'emergency',now):raise ValueError('res_emergency')
            ready=value(signals,'ready',now)
            go=value(signals,'go',now)
            if ready and self.ready_since is None:self.ready_since=now
            if not ready:
                self.ready_since=None
                if self.go_accepted:raise ValueError('readiness_lost')
            if go and not self.last_go and self.ready_since is not None and now-self.ready_since>=5.:
                self.go_accepted=True
            self.last_go=go
            permitted=self.go_accepted and ready and value(signals,'driving',now) and value(signals,'sound_done',now)
            return dict(permitted=bool(permitted),reason=None,vehicle_output=False)
        except ValueError as exc:
            reason=str(exc)
            if not self.go_accepted and (reason.startswith('stale_') or reason.startswith('missing_')):
                self.ready_since=None
                self.last_go=True # Recovery must observe low before a fresh rising edge.
                return dict(permitted=False,reason=reason,waiting=True,vehicle_output=False)
            self.fault=reason
            return dict(permitted=False,reason=self.fault,waiting=False,vehicle_output=False)

class FinishController(object):
    def __init__(self,front,rear,half_width,brake,latency,margin=.5,target=10.,dwell=.5):
        if min(front,rear,half_width,brake,margin,target,dwell)<=0 or latency<0 or target+front+margin>=30:
            raise ValueError('invalid_finish_configuration')
        self.front,self.rear,self.half_width=front,rear,half_width
        self.brake,self.latency,self.margin=brake,latency,margin
        self.target,self.dwell=target,dwell;self.still_since=None;self.fault=None;self.last=None
    def step(self,now,after,speed,yaw_error,signals):
        try:
            if self.fault:raise ValueError(self.fault)
            if not all(not math.isnan(v) and not math.isinf(v) for v in (now,after,speed,yaw_error)) or speed<0:raise ValueError('invalid_finish_input')
            if self.last is not None and now<=self.last:raise ValueError('finish_time_rewind')
            self.last=now
            if value(signals,'emergency',now):raise ValueError('res_emergency')
            if abs(value(signals,'speed',now)-speed)>.01:raise ValueError('speed_feedback_mismatch')
            # Support of an IMU-referenced rectangle along finish tangent.
            c,s=math.cos(yaw_error),abs(math.sin(yaw_error))
            leading=max(self.front*c,-self.rear*c)+self.half_width*s
            trailing=min(self.front*c,-self.rear*c)-self.half_width*s
            remaining=30.-after-leading-self.margin
            if remaining<0:raise ValueError('whole_body_stop_zone_exceeded')
            available=min(remaining,max(0.,self.target-after))
            cap=max(0.,math.sqrt((self.brake*self.latency)**2+2*self.brake*available)-self.brake*self.latency)
            required=speed*self.latency+speed*speed/(2*self.brake)
            if required>remaining:raise ValueError('insufficient_braking_distance')
            inside=after+trailing>=0
            if speed<=.02 and inside:
                if self.still_since is None:self.still_since=now
            else:self.still_since=None
            complete=self.still_since is not None and now-self.still_since>=self.dwell
            ack=False
            if complete:
                ack=bool(value(signals,'as_finished',now) and value(signals,'ebs_active',now) and value(signals,'drive_disabled',now))
            return dict(phase='FINISHED_CONFIRMED' if ack else 'COMPLETE_REQUEST' if complete else 'STOPPING',
                        completion_request=complete,hardware_confirmed=ack,speed_limit=0. if complete else cap,
                        remaining=remaining,leading_extent=leading,reason=None,vehicle_output=False)
        except ValueError as exc:
            self.fault=str(exc)
            return dict(phase='FAULT',completion_request=False,hardware_confirmed=False,speed_limit=0.,reason=self.fault,vehicle_output=False)
