"""Offline Trackdrive mission supervisor, NOT the hardware AS state machine.

Requires trusted map progress, directed gate crossing and hardware status adapters.
No CAN/UDP, ASSI, SDC or hardware AS command is implemented.
"""
import math

class TrackdriveMission(object):
    def __init__(self, length, max_speed, brake, latency, laps=3):
        if min(length,max_speed,brake)>0 and latency>=0 and laps==3:
            self.length,self.max_speed,self.brake,self.latency=length,max_speed,brake,latency
        else:raise ValueError('invalid_mission_configuration')
        self.state='WAIT_READY' # Internal task phase only, never published as AS state.
        self.ready_since=None
        self.last=None
        self.start_time=None
        self.lap_time=None
        self.distance=0.
        self.lap_origin=0.
        self.laps=0
        self.warnings=[]
        self.reason=None
        self.finish_distance=None

    def step(self, now, progress, speed, gate=False, ready=False, go=False,
             driving_ack=False, sound_done=False, emergency=False, healthy=True):
        out=dict(vehicle_output=False,motion_permitted=False,speed_limit=0.)
        try:
            if not all(not math.isnan(v) and not math.isinf(v) for v in (now,progress,speed)):
                raise ValueError('nonfinite_mission_input')
            if not 0<=progress<self.length or speed<0:raise ValueError('invalid_progress_or_speed')
            if self.state=='FAULT':raise ValueError(self.reason)
            if emergency or not healthy:raise ValueError('emergency_or_unhealthy')
            if self.last is not None and now<=self.last[0]:raise ValueError('non_monotonic_time')
            if self.state=='WAIT_READY':
                if not ready:self.ready_since=None
                elif self.ready_since is None:self.ready_since=now
                if go and self.ready_since is not None and now-self.ready_since>=5.:
                    self.state='WAIT_DRIVING_ACK'
            if self.state=='WAIT_DRIVING_ACK':
                if not ready:raise ValueError('readiness_lost')
                if driving_ack and sound_done:self.state='SEEK_START'
            if self.state in ('SEEK_START','LAPPING','STOPPING'):
                if not ready or not driving_ack or not sound_done:raise ValueError('hardware_permission_lost')
                delta=0.
                wrap=False
                if self.last is not None:
                    dt=now-self.last[0]
                    raw=progress-self.last[1]
                    wrap=raw < -self.length/2.
                    delta=raw+self.length if wrap else raw
                    if delta < -.05 or delta>self.max_speed*dt+.5:raise ValueError('progress_jump_or_reverse')
                    delta=max(0.,delta)
                    if wrap and not gate:raise ValueError('unconfirmed_finish_gate')
                self.distance+=delta
                if self.state=='SEEK_START' and gate:
                    self.state='LAPPING';self.start_time=self.lap_time=now;self.lap_origin=self.distance
                elif self.state=='LAPPING' and gate and wrap:
                    covered=self.distance-self.lap_origin
                    if covered<.9*self.length:raise ValueError('insufficient_lap_progress')
                    duration=now-self.lap_time
                    average=covered/duration
                    self.laps+=1
                    threshold=.5 if self.laps==1 else 2.
                    if average<threshold:self.warnings.append(dict(lap=self.laps,average=average,threshold=threshold))
                    self.lap_time=now;self.lap_origin=self.distance
                    if self.laps==3:
                        self.state='STOPPING';self.finish_distance=self.distance
                if self.state=='STOPPING':
                    after=self.distance-self.finish_distance
                    # Conservative leading body/uncertainty allowance must be supplied
                    # by a future vehicle adapter; this core uses reference-point distance.
                    remaining=30.-after
                    if remaining<0:raise ValueError('stop_zone_exceeded')
                    out['speed_limit']=max(0.,math.sqrt((self.brake*self.latency)**2+2*self.brake*remaining)-self.brake*self.latency)
                    if speed<=.02:self.state='COMPLETE_CANDIDATE'
                else:out['speed_limit']=self.max_speed
                out['motion_permitted']=self.state!='COMPLETE_CANDIDATE'
            self.last=(now,progress)
        except ValueError as exc:
            self.state='FAULT';self.reason=str(exc)
            out.update(motion_permitted=False,speed_limit=0.)
        out.update(phase=self.state,laps=self.laps,reason=self.reason,warnings=list(self.warnings))
        return out


class TrackdriveSupervisor(object):
    """Strict candidate coordinator. Legacy bool/progress authorization is rejected."""
    def __init__(self, planner, mission, progress_tracker=None, finish_controller=None):
        from .finish import StartupGate
        self.planner,self.mission=planner,mission
        self.progress_tracker,self.finish_controller=progress_tracker,finish_controller
        self.startup=StartupGate();self.fault=None;self.finish_origin=None

    def step(self,scene,now,progress=None,gate=False,signals=None,
             finish_yaw_error=None,stop_geometry_confirmed=False,**legacy):
        from .finish import value
        result=dict(candidate_valid=False,speed_limit=0.,steering=None,vehicle_output=False,reason=None)
        try:
            if self.fault:raise ValueError(self.fault)
            if legacy or progress is not None or gate:raise ValueError('legacy_authorization_rejected')
            if signals is None:raise ValueError('timestamped_signals_required')
            if self.progress_tracker is None or self.finish_controller is None:raise ValueError('task_adapters_required')
            # Validate speed and hardware timestamps even before requesting motion.
            speed=value(signals,'speed',now)
            if scene.feedback is None or not 0<=now-scene.feedback[2]<=.2:raise ValueError('scene_feedback_stale')
            if abs(scene.feedback[0]-speed)>.01:raise ValueError('speed_feedback_mismatch')
            if not 0<=now-scene.stamp<=.5:raise ValueError('pose_observation_stale')
            startup=self.startup.step(now,signals)
            if startup['reason']:raise ValueError(startup['reason'])
            observation=self.progress_tracker.update(scene.pose,scene.stamp,scene.version)
            planning=self.planner.step(scene,now,authorized=False)
            if not planning['valid']:raise ValueError(planning['reason'])
            mission=self.mission.step(now,observation['progress'],speed,gate=observation['gate'],
                ready=value(signals,'ready',now),go=self.startup.go_accepted,
                driving_ack=value(signals,'driving',now),sound_done=value(signals,'sound_done',now))
            result.update(planning=planning,mission=mission,progress=observation)
            if mission['phase']=='FAULT':raise ValueError(mission['reason'])
            cap=min(planning['speed_limit'],mission['speed_limit'])
            permitted=startup['permitted'] and mission['motion_permitted']
            if mission['phase'] in ('STOPPING','COMPLETE_CANDIDATE'):
                if self.finish_origin is None:self.finish_origin=observation['unwrapped_progress']
                if not stop_geometry_confirmed or finish_yaw_error is None:raise ValueError('stop_geometry_unconfirmed')
                finish=self.finish_controller.step(now,observation['unwrapped_progress']-self.finish_origin,
                                                   speed,finish_yaw_error,signals)
                if finish['phase']=='FAULT':raise ValueError(finish['reason'])
                cap=min(planning['speed_limit'],finish['speed_limit'])
                permitted=startup['permitted'] and not finish['completion_request']
                result['finish']=finish
            result.update(planning=planning,mission=mission,progress=observation,
                          candidate_valid=bool(permitted),speed_limit=cap if permitted else 0.,
                          steering=planning['steering'] if permitted else None)
        except (ValueError,TypeError,KeyError,IndexError) as exc:
            reason=str(exc)
            recoverable=(reason.startswith('stale_') or reason.startswith('missing_') or reason in (
                'timestamped_signals_required','scene_feedback_stale','pose_observation_stale',
                'no_start_gate','insufficient_corridor','free_space_unavailable',
                'feedback_unavailable','observation_stale','vehicle_parameters_unavailable'))
            if not self.startup.go_accepted and self.fault is None and recoverable:
                self.startup.ready_since=None
                self.startup.last_go=True
                if self.mission.state=='WAIT_READY':self.mission.ready_since=None
                result.update(reason=reason,waiting=True,fault_latched=False)
            else:
                self.fault=reason
                result.update(reason=reason,waiting=False,fault_latched=True)
        return result
