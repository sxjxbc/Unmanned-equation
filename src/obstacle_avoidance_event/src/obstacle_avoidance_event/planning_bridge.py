"""Read-only message-to-planner bridge. All execution requests stay in memory."""
import math
from .input_adapter import AlignedIngress
from .legacy_input import LegacyCoordinates
from .compatibility import LegacyCompatibleRunner
from .corridor import Corridor
from .execution_adapter import finite
from .pipeline import REQUIRED
from .boundary_adapter import infer_boundary


def configured_ingress(config):
    if config.get('localization_reference')!='confirmed_imu':
        raise ValueError('localization_reference_unconfirmed')
    converter=LegacyCoordinates(config['lidar_offset_x'],config['lidar_offset_y'],
        config['lidar_y_sign'],config['lidar_yaw_offset_deg'],
        rear_in_imu=config.get('rear_in_imu'),convention_confirmed=config.get('convention_confirmed') is True)
    def pose_transform(p):
        return converter.pose_to_rear(dict(current_x=p[0],current_y=p[1],
                                           current_yaw=math.degrees(p[2]),origin_yaw=0.))
    return AlignedIngress(map_frame=config.get('map_frame','trackdrive_map'),
        input_map_frame=config.get('input_map_frame','map'),
        cones_frame=config.get('cones_frame','/velodyne'),pose_reference='rear_axle',
        pose_transform=pose_transform,point_transform=converter.lidar_to_rear)


class PlanningShadowBridge(object):
    def __init__(self,ingress=None,vehicle=None,synthetic=False,configuration_error=None,steering_sign=None,
                 boundary_mode='external_confirmed',trial_permission=False,ready_frames=10,candidate_hold=.2):
        if boundary_mode not in ('external_confirmed','inferred_shadow','inferred_low_speed_trial'):raise ValueError('invalid_boundary_mode')
        if ready_frames<2 or not finite(candidate_hold) or not 0<candidate_hold<=.5:raise ValueError('invalid_readiness_configuration')
        self.trial_permission=trial_permission is True
        self.ready_frames,self.valid_frames,self.candidate_hold=ready_frames,0,candidate_hold
        self.boundary_mode=boundary_mode
        self.ingress=ingress
        self.error=configuration_error
        self.runner=None
        if vehicle is not None:
            if any(not finite(vehicle.get(k)) or vehicle[k]<=0 for k in REQUIRED):
                self.error='vehicle_parameters_unavailable'
            elif vehicle.get('pose_reference')!='rear_axle':
                self.error='vehicle_rear_reference_unconfirmed'
            elif steering_sign not in (-1,1):
                self.error='execution_steering_sign_unconfirmed'
            else:
                self.runner=LegacyCompatibleRunner(vehicle,steering_sign,synthetic=synthetic,
                    allow_inferred_shadow=boundary_mode in ('inferred_shadow','inferred_low_speed_trial'))
        self.cones=None;self.boundary=None;self.input_error=None
        self.processed=None;self.last_result=None;self.last_stamp=None

    def add_pose(self,message):
        if self.ingress is None:return
        try:
            self.ingress.add_pose(message)
            if self.input_error and self.input_error.startswith('pose:'):self.input_error=None
        except (ValueError,TypeError,AttributeError) as exc:
            self.ingress.poses.samples=[]
            self.input_error='pose:'+str(exc)

    def add_cones(self,message):
        self.cones=message

    def add_boundary(self,packet):
        """Internal JSON contract, not an assumed existing sensor topic.

        Ordered left/right points must already be in the NEW map frame.
        Confirmation asserts upstream boundary identity, not obstacle absence.
        """
        try:
            if packet.get('confirmed') is not True:raise ValueError('boundary_identity_unconfirmed')
            if self.ingress is None:raise ValueError('coordinate_adapter_unavailable')
            if packet.get('frame')!=self.ingress.map_frame:raise ValueError('boundary_frame_mismatch')
            stamp=packet['stamp']
            if not finite(stamp) or stamp<=0:raise ValueError('invalid_boundary_stamp')
            if self.boundary and stamp<self.boundary['stamp']:raise ValueError('boundary_time_rewind')
            left,right=packet['left'],packet['right']
            if len(left)+len(right)>240 or any(len(p)!=2 or not all(finite(v) for v in p) for p in list(left)+list(right)):
                raise ValueError('invalid_boundary_points')
            corridor=Corridor(left,right,confirmed=True)
            source=packet.get('source')
            allowed=('synthetic',) if self.runner and self.runner.assembler.synthetic else ('confirmed_boundary','certified_sensor')
            if self.boundary_mode=='inferred_shadow':allowed+=('inferred_boundary_shadow_only',)
            if self.boundary_mode=='inferred_low_speed_trial':allowed+=('inferred_boundary_low_speed_trial',)
            if source not in allowed:raise ValueError('boundary_source_unconfirmed')
            self.boundary=dict(stamp=stamp,frame=packet['frame'],version=packet['version'],
                               source=source,free=corridor.free,obstacles=packet.get('obstacles',[]))
            if self.input_error and self.input_error.startswith('boundary:'):self.input_error=None
        except (ValueError,KeyError,TypeError,AttributeError) as exc:
            self.boundary=None
            self.input_error='boundary:'+str(exc)

    def unavailable(self,reason,now,allow_hold=False):
        if allow_hold and self.last_result and self.last_result.get('candidate_valid') and self.last_stamp is not None:
            if finite(now) and 0<=now-self.last_stamp<=self.candidate_hold:
                out=dict(self.last_result);out['held']=True
                out['execution']=dict(out['execution'],request_recorded=False)
                return out
        self.valid_frames=0
        out=dict(candidate_valid=False,vehicle_output=False,reason=reason,waiting=True,
                 control_mode='legacy_compatible',actual_speed=None,actual_steering=None,
                 hardware_go=None,hardware_as_state=None,
                 execution=dict(action='wait',request_recorded=False,vehicle_output=False))
        if self.runner and (self.runner.execution.started or self.runner.fault):
            self.runner.fault=self.runner.fault or reason
            out.update(waiting=False,fault_latched=True,reason=self.runner.fault)
            out['execution']=self.runner.execution.step(out,self.last_stamp,now)
        return out

    def tick(self,now):
        if self.error:return self.unavailable(self.error,now)
        if self.ingress is None:return self.unavailable('coordinate_adapter_unavailable',now)
        if self.runner is None:return self.unavailable('vehicle_parameters_unavailable',now)
        if self.runner.fault:return self.unavailable(self.runner.fault,now)
        if self.input_error:return self.unavailable(self.input_error,now)
        if self.cones is None:return self.unavailable('cones_unavailable',now,True)
        if self.boundary is None and self.boundary_mode=='external_confirmed':
            return self.unavailable('confirmed_boundary_unavailable',now,True)
        try:
            aligned=self.ingress.align(self.cones,now)
            stamp=aligned['stamp']
            if self.boundary_mode in ('inferred_shadow','inferred_low_speed_trial') and stamp!=self.processed:
                vehicle=self.runner.planner.vehicle
                clearance=(vehicle['rear']+math.hypot(.125,vehicle['width']/2.)+vehicle['margin']+.2
                           if self.boundary_mode=='inferred_low_speed_trial' else 0.)
                packet=infer_boundary(aligned,previous=self.runner.planner.previous,rear_clearance=clearance)
                if self.boundary_mode=='inferred_low_speed_trial':packet['source']='inferred_boundary_low_speed_trial'
                self.add_boundary(packet)
                if self.input_error:return self.unavailable(self.input_error,now)
                # Keep the inferred far cap inside observations. Shorten the
                # planning horizon rather than invent free space past the cap.
                left,right=packet['left'],packet['right']
                end=((left[-1][0]+right[-1][0])/2.,(left[-1][1]+right[-1][1])/2.)
                prior=((left[-2][0]+right[-2][0])/2.,(left[-2][1]+right[-2][1])/2.)
                heading=math.atan2(end[1]-prior[1],end[0]-prior[0])
                vehicle=self.runner.planner.vehicle
                reserve=math.hypot(max(vehicle['front'],vehicle['rear']),vehicle['width']/2.)+vehicle['margin']+.3
                retained=[];withheld=[]
                for point in aligned['cones']:
                    along=(point[0]-end[0])*math.cos(heading)+(point[1]-end[1])*math.sin(heading)
                    (retained if along<=-reserve else withheld).append(point)
                aligned['cones']=retained
                aligned['obstacles']=aligned.get('obstacles',[])+[(p[0],p[1],.15) for p in withheld]
            if abs(self.boundary['stamp']-stamp)>.001:
                return self.unavailable('boundary_observation_time_mismatch',now,True)
            if self.processed is not None and stamp==self.processed:
                if self.last_result and not self.runner.execution.started:return dict(self.last_result)
                return self.unavailable('candidate_stale',now,True)
            aligned['obstacles']=aligned.get('obstacles',[])+self.boundary['obstacles']
            authorized=(self.trial_permission and self.valid_frames+1>=self.ready_frames) if self.boundary_mode=='inferred_low_speed_trial' else True
            result=self.runner.step(aligned,self.boundary,now,version=self.boundary['version'],motion_authorized=authorized)
            self.valid_frames=self.valid_frames+1 if result.get('candidate_valid') else 0
            eligible=(self.boundary_mode=='inferred_low_speed_trial' and self.trial_permission
                and self.runner.planner.vehicle.get('geometry_mode')=='rear_axle_rectangle'
                and self.valid_frames>=self.ready_frames and result.get('candidate_valid') is True)
            result.update(boundary_source=self.boundary['source'],drive_eligible=bool(eligible),
                          trial_authorized=self.trial_permission,valid_frames=self.valid_frames,
                          boundary_certified=False)
            self.processed=stamp;self.last_stamp=stamp;self.last_result=result
            return result
        except (ValueError,KeyError,TypeError,AttributeError) as exc:
            reason=str(exc)
            return self.unavailable(reason,now,allow_hold=reason=='pose_bracket_unavailable')
