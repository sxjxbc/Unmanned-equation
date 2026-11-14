"""Local-only command arbitration. No vehicle transport or feedback synthesis."""
import math


def finite(value):
    return isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value)


class RecordingClient(object):
    """Same call signature as legacy CANUDPClient, records requests in memory."""
    def __init__(self):
        self.requests = []

    def send_steering_frame(self, steering_angle, stop=False):
        self.requests.append(dict(steering_angle=steering_angle, stop=stop))
        return True


class LocalExecutionAdapter(object):
    """Consume strict Supervisor candidates; intentionally cannot enable output.

    Legacy steering_sign is explicit because internal left-positive geometry
    does not establish the physical actuator sign. Waiting emits no request;
    an explicit emergency, fault or stop emits the legacy stop call signature.
    Once motion has been requested, loss of a valid candidate latches stop.
    A new instance is required after a latched stop.
    """
    def __init__(self, steering_sign, max_steer, max_age=.2, vehicle_output=False,
                 max_steering_step=None):
        if vehicle_output:
            raise ValueError('vehicle_output_not_supported')
        if steering_sign not in (-1, 1) or not finite(max_steer) or not 0 < max_steer <= math.pi/2:
            raise ValueError('invalid_execution_configuration')
        if not finite(max_age) or max_age <= 0:
            raise ValueError('invalid_execution_configuration')
        self.sign, self.max_steer, self.max_age = steering_sign, max_steer, max_age
        self.client = RecordingClient()
        if max_steering_step is not None and (not finite(max_steering_step) or max_steering_step<=0):
            raise ValueError('invalid_steering_step')
        self.max_step=max_steering_step
        self.previous_steering=0.
        self.started = False
        self.latched = None
        self.last_now = None

    def step(self, candidate, source_stamp, now, emergency=False, stop_requested=False):
        result = dict(vehicle_output=False, action='wait', reason=None,
                      steering=None, speed_limit=0., request_recorded=False)
        reason = self.latched
        if not finite(now) or (self.last_now is not None and now < self.last_now):
            reason = reason or 'invalid_execution_time'
        elif emergency or stop_requested:
            reason = reason or ('emergency' if emergency else 'stop_requested')
        elif not isinstance(candidate, dict):
            reason = reason or 'invalid_candidate'
        elif candidate.get('fault_latched'):
            reason = reason or candidate.get('reason') or 'planner_fault'
        elif candidate.get('candidate_valid') is not True:
            if self.started:
                reason = reason or 'candidate_permission_lost'
        else:
            angle, speed = candidate.get('steering'), candidate.get('speed_limit')
            if not finite(source_stamp) or not 0 <= now-source_stamp <= self.max_age:
                reason = reason or 'candidate_stale'
            elif not finite(angle) or abs(angle) > self.max_steer:
                reason = reason or 'invalid_candidate_steering'
            elif not finite(speed) or speed < 0:
                reason = reason or 'invalid_candidate_speed'
            elif speed == 0:
                reason = reason or 'zero_speed_stop'
            elif reason is None:
                angle *= self.sign
                result['requested_steering']=angle
                if self.max_step is not None:
                    angle=max(self.previous_steering-self.max_step,min(self.previous_steering+self.max_step,angle))
                self.client.send_steering_frame(angle, stop=False)
                self.previous_steering=angle
                self.started = True
                result.update(action='steer', steering=angle, speed_limit=speed,
                              request_recorded=True)
        if reason is not None:
            self.latched = reason
            self.client.send_steering_frame(0., stop=True)
            result.update(action='stop', reason=reason, request_recorded=True)
        if finite(now):
            self.last_now = now
        return result
