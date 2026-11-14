import math
import copy
from .model import fresh, distance, wrap
from .gates import gates
from .search import search, oriented_gate
from .association import ConeTracker
from .trajectory import spline, validate, select_path
from .control import pursuit, speed_limit
from .execution_adapter import finite

REQUIRED = ('wheelbase','front','rear','width','margin','max_steer','steer_rate',
            'speed_cap','lateral_accel','brake','latency','stop_margin')

class Pipeline(object):
    def __init__(self, vehicle=None, mode='strict'):
        if mode not in ('strict','legacy_compatible'):raise ValueError('invalid_control_mode')
        self.mode = mode
        self.vehicle = vehicle
        self.state = 'WAIT_INPUT'
        self.previous = None
        self.version = None
        self.last_now = None
        self.tracker = ConeTracker()
        self.side_labels = {}
        self.gate_history = []
        self.last_pose = None
        self.committed_path = None
        self.committed_offset = 0.
        self.maneuver_obstacles = []

    def step(self, scene, now, authorized=False):
        result = {'state':self.state,'valid':False,'reason':None,'path':[],
                  'steering':0.,'speed_limit':0.,'vehicle_output':False,
                  'control_mode':self.mode,'actual_steering':None,
                  'hardware_state_source':'not_integrated'}
        try:
            if self.state == 'STOP_LATCHED':
                raise ValueError('stop_latched')
            if self.last_now is not None and now < self.last_now:
                raise ValueError('clock_rewind')
            self.last_now = now
            if self.vehicle and self.vehicle.get('pose_reference') == 'imu':
                raise ValueError('imu_profile_requires_rear_axle_adapter')
            if not self.vehicle or any(k not in self.vehicle or not isinstance(self.vehicle[k], (int,float)) or math.isnan(self.vehicle[k]) or math.isinf(self.vehicle[k]) or self.vehicle[k] <= 0 for k in REQUIRED):
                raise ValueError('vehicle_parameters_unavailable')
            if not all(not math.isnan(v) and not math.isinf(v) for v in scene.pose):
                raise ValueError('invalid_pose')
            if not all(not math.isnan(v) and not math.isinf(v) for p in scene.cones for v in p):
                raise ValueError('invalid_cone')
            if not fresh(scene.stamp,now,.5):
                raise ValueError('observation_stale')
            if self.mode == 'strict':
                if not scene.feedback or not fresh(scene.feedback[2],now,.2):
                    raise ValueError('feedback_unavailable')
                if any(math.isnan(v) or math.isinf(v) for v in scene.feedback):
                    raise ValueError('invalid_feedback')
                speed,measured_steer=scene.feedback[:2]
                result['actual_steering']=measured_steer
            else:
                estimate=scene.speed_estimate
                if not estimate or estimate.get('source')!='GPS_pose_difference_filtered_clipped_not_wheel_speed':
                    raise ValueError('pose_speed_estimate_unavailable')
                speed=estimate.get('speed')
                if not isinstance(speed,(int,float)) or math.isnan(speed) or math.isinf(speed):
                    raise ValueError('invalid_speed_estimate')
                if not finite(estimate.get('stamp')) or not fresh(estimate.get('stamp'),now,.8):
                    raise ValueError('speed_estimate_stale')
                measured_steer=None
                result.update(estimated_speed=speed,speed_source=estimate['source'],
                              steering_response_check='unavailable_external_CAN_monitoring')
            if self.version is not None and scene.version != self.version:
                raise ValueError('map_version_changed')
            if scene.free is None:
                raise ValueError('free_space_unavailable')
            # Work on a copy: rejected paths must not advance map identities/history.
            tracker = copy.deepcopy(self.tracker)
            tracked = tracker.update(scene.cones,scene.stamp,scene.version)
            ordered_ids = sorted(tracked)
            ordered_points = [tracked[i] for i in ordered_ids]
            gs,adj = gates(ordered_points)
            historical_labels = dict((tracked[i],label) for i,label in self.side_labels.items() if i in tracked)
            centers,ids = search(gs,adj,scene.pose,self.previous,initial_labels=historical_labels)
            stable_gates = [tuple(sorted((ordered_ids[a],ordered_ids[b]))) for a,b in ids]
            if self.gate_history:
                positions = dict((g,i) for i,g in enumerate(self.gate_history))
                overlap = [positions[g] for g in stable_gates if g in positions]
                if len(overlap) < 2:
                    raise ValueError('route_identity_lost')
                if any(b <= a for a,b in zip(overlap,overlap[1:])):
                    raise ValueError('route_order_reversed')
                if overlap[0] > 3:
                    raise ValueError('route_progress_jump')
            if self.last_pose is not None:
                dt = scene.stamp-self.tracker.stamp
                if distance(scene.pose[:2],self.last_pose[:2]) > self.vehicle['speed_cap']*dt+.4:
                    raise ValueError('pose_progress_jump')
            headings = [math.atan2(b[1]-a[1],b[0]-a[0]) for a,b in zip(centers,centers[1:])]
            total_turn = sum(abs(wrap(b-a)) for a,b in zip(headings,headings[1:]))
            if total_turn > 1.2:
                raise ValueError('tight_turn_topology_unsupported')
            # Avoid a vanishing first Hermite segment when crossing a gate.
            # Keep topology/history intact; only omit near anchors for fitting.
            fitting_centers = [p for p in centers if distance(p,scene.pose[:2]) >= .8]
            if len(fitting_centers) < 2:
                raise ValueError('insufficient_fitting_horizon')
            path = spline([scene.pose[:2]]+fitting_centers)
            release = False
            if self.committed_path is not None:
                # Only release after the original obstacles are behind the body.
                x,y,yaw = scene.pose
                behind = bool(self.maneuver_obstacles) and all(
                    math.cos(yaw)*(ox-x)+math.sin(yaw)*(oy-y) < -(self.vehicle['rear']+radius+.3)
                    for ox,oy,radius in self.maneuver_obstacles)
                lateral = float('inf')
                tangent = yaw
                # Ordered current corridor segment, not the path starting at ego.
                for a,b in zip(centers,centers[1:]):
                    dx,dy=b[0]-a[0],b[1]-a[1]
                    t=max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/max(dx*dx+dy*dy,1e-9)))
                    d=distance((x,y),(a[0]+t*dx,a[1]+t*dy))
                    if d<lateral:lateral,tangent=d,math.atan2(dy,dx)
                release = behind and lateral < .12 and abs(wrap(yaw-tangent)) < .15
            if self.committed_path is not None and not release:
                # Keep a validated maneuver in fixed map coordinates. Recheck
                # against CURRENT observations; never blindly replay old safety.
                prior = self.committed_path
                nearest = min(range(min(12,len(prior))),key=lambda i:distance(prior[i],scene.pose[:2]))
                if distance(prior[nearest],scene.pose[:2]) > .5:
                    raise ValueError('committed_path_tracking_lost')
                path = prior[nearest:]
                if len(path) < 3:
                    raise ValueError('committed_path_horizon_exhausted')
                ks = validate(path,scene,self.vehicle)
                offset = self.committed_offset
            else:
                path,ks,offset = select_path(path,scene,self.vehicle)
            steer,progress = pursuit(path,scene.pose,self.vehicle)
            cap = speed_limit(path,ks,self.vehicle,measured_steer,steer,
                              check_response=self.mode=='strict')
            if speed < 0 or speed > cap+.05:
                raise ValueError('speed_outside_envelope')
            labels = dict(self.side_labels)
            heading = scene.pose[2]
            coord_ids = dict((p,i) for i,p in tracked.items())
            for index,e in enumerate(ids):
                if index:
                    a,b = centers[index-1],centers[index]
                    heading = math.atan2(b[1]-a[1],b[0]-a[0])
                sides = oriented_gate(gs[e],heading)
                if sides is None:
                    raise ValueError('invalid_selected_gate')
                for point,label in zip(sides,('left','right')):
                    identity = coord_ids[point]
                    if identity in labels and labels[identity] != label:
                        raise ValueError('historical_side_conflict')
                    labels[identity] = label
            if release:
                self.committed_path = None
                self.committed_offset = 0.
                self.maneuver_obstacles = []
            if offset != 0.:
                if self.committed_path is None:
                    self.maneuver_obstacles = list(scene.obstacles)
                self.committed_path = list(path)
                self.committed_offset = offset
            self.state = 'RUN' if authorized else 'READY'
            self.tracker,self.side_labels = tracker,dict((i,v) for i,v in labels.items() if i in tracked)
            self.gate_history,self.last_pose = stable_gates,scene.pose
            self.previous,self.version = centers,scene.version
            result.update(valid=True,path=path,steering=steer,speed_limit=cap,
                          maneuver_released=release,selected_offset=offset,gate_sequence=ids,stable_gate_sequence=stable_gates,cone_ids=ordered_ids,curvature=ks,map_version=scene.version,
                          source_stamp=scene.stamp)
        except (ValueError,ArithmeticError) as exc:
            result['reason'] = str(exc)
            if self.state == 'RUN':
                self.state = 'STOP_LATCHED'
        result['state'] = self.state
        return result
