#!/usr/bin/env python3
"""Actual legacy planner + complete controller, in-memory transport and assumed plant.

No ROS, sockets, hardware or third-party packages. Map/pose inputs are synthetic;
GPS driver, mapping, real ROS timing and vehicle actuator calibration are excluded.
"""
import argparse
from collections import deque
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

PACKAGE = Path(__file__).resolve().parents[1]


class Stamp:
    def __init__(self, value=0.0):
        self.value = value

    def to_sec(self):
        return self.value

    def is_zero(self):
        return self.value == 0

    def __sub__(self, other):
        return Stamp(self.value-other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value


class Header:
    def __init__(self):
        self.stamp, self.frame_id = Stamp(), ''


class Pose:
    def __init__(self):
        self.position = types.SimpleNamespace(x=0., y=0., z=0.)
        self.orientation = types.SimpleNamespace(x=0., y=0., z=0., w=1.)


class PoseStamped:
    def __init__(self):
        self.header, self.pose = Header(), Pose()


class Points:
    def __init__(self):
        self.header, self.poses, self.cones = Header(), [], []


class Scalar:
    def __init__(self, data=None):
        self.data = data


class Imu:
    def __init__(self):
        self.header = Header()
        self.orientation = types.SimpleNamespace(z=0.)


class ImportOnlyNumpy(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError('Unexpected NumPy computation in harness: '+name)


class Harness:
    def __init__(self, parameter_overrides=None):
        self.now = 100.
        self.node = ''
        self.callbacks, self.latest, self.messages, self.errors = {}, {}, {}, []
        self.stop_events = []
        self.timer_ticks = []
        self.samples = []
        launch = ET.parse(str(PACKAGE/'launch'/'acceleration_event.launch')).getroot()
        self.params = {}
        for node in launch.findall('node'):
            values = {}
            for item in node.findall('param'):
                value = item.get('value')
                if value.startswith('$(arg '):
                    continue
                try:
                    value = json.loads(value.lower())
                except (ValueError, TypeError):
                    pass
                values['~'+item.get('name')] = value
            self.params[node.get('name')] = values
        self.params['path_tracking_controller']['~enable_vehicle_output'] = False
        if parameter_overrides:
            self.params['path_tracking_controller'].update(parameter_overrides)
        # Offline experiments must never override the output isolation.
        self.params['path_tracking_controller']['~enable_vehicle_output'] = False
        self._load()

    def _load(self):
        saved = dict(sys.modules)
        old_path = list(sys.path)
        def module(name, **attrs):
            item = types.ModuleType(name)
            item.__dict__.update(attrs)
            sys.modules[name] = item
            return item
        def init_node(name, **kwargs):
            self.node = name
        def subscribe(topic, kind, callback, **kwargs):
            self.callbacks.setdefault(topic, []).append((self.node, callback))
        def publisher(topic, *args, **kwargs):
            return types.SimpleNamespace(publish=lambda msg: self.publish(topic, msg))
        def timer(duration, callback, **kwargs):
            self.timer_ticks.append(callback)
            return types.SimpleNamespace(shutdown=lambda:None)
        def error(*args):
            # Alarm-level messages during expected stale-road hold are evidence,
            # not Python execution failures. Actual exceptions propagate or are
            # reported by the controller as "Legacy control error".
            destination = self.errors if any('Legacy control error' in str(a) for a in args) else self.stop_events
            destination.append(str(args))
        noop = lambda *a, **kw:None
        time_class = type('Time', (Stamp,), {'now':staticmethod(lambda:Stamp(self.now))})
        ros = module('rospy', init_node=init_node, Publisher=publisher, Subscriber=subscribe,
                     get_param=lambda key, default=None:self.params.get(self.node, {}).get(key, default),
                     Time=time_class, Duration=lambda value:value, Timer=timer,
                     loginfo=noop,logwarn=noop,logdebug_throttle=noop,loginfo_throttle=noop,
                     logwarn_throttle=noop,logdebug=noop,logwarn_once=noop,
                     logerr=error,logerr_throttle=error)
        module('numpy')
        sys.modules['numpy'] = ImportOnlyNumpy('numpy')
        for name, classes in (
                ('geometry_msgs',dict(PoseStamped=PoseStamped,PoseArray=Points)),
                ('nav_msgs',dict(Path=Points)),('mapping',dict(ConeArray=Points)),
                ('std_msgs',dict(Header=Header,Float64=Scalar,String=Scalar)),
                ('sensor_msgs',dict(Imu=Imu))):
            package = module(name)
            package.msg = module(name+'.msg', **classes)
        tf = module('tf')
        tf.transformations = module('tf.transformations', euler_from_quaternion=lambda q:
            (0.,0.,math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))))
        # Deterministic legacy SLAM mode; no simulator package discovery.
        module('fssim_common')
        sys.modules.pop('fssim_common.msg', None)
        sys.path.insert(0,str(PACKAGE/'src'))
        def load(filename):
            name = '_readiness_'+filename
            spec = importlib.util.spec_from_file_location(name, PACKAGE/'src'/(filename+'.py'))
            item = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(item)
            item.time = types.SimpleNamespace(time=lambda:self.now)
            return item
        try:
            planning, control = load('zhixian_realtime'), load('genzong_realtime')
            control.ControlLogger = lambda folder:types.SimpleNamespace(
                log_control_data=lambda record:self.samples.append(dict(record)), close=noop)
            def forbidden():
                raise AssertionError('CAN constructor called in disabled validation mode')
            control.CANUDPClient = forbidden
            self.planner = planning.RealtimeStraightLinePlanner()
            self.controller = control.PathTrackingController()
            if self.controller.can_client is not None or self.controller.cmd_pub is not None:
                raise AssertionError('Output resources created')
        finally:
            sys.path[:] = old_path
            for name in list(sys.modules):
                if name not in saved:
                    del sys.modules[name]
            sys.modules.update(saved)

    def publish(self, topic, msg):
        if topic in ('/control/steering','/fssim/cmd'):
            raise AssertionError('Vehicle command topic used: '+topic)
        self.latest[topic] = msg
        self.messages[topic] = self.messages.get(topic,0)+1
        old = self.node
        try:
            for scope, callback in self.callbacks.get(topic, []):
                self.node = scope
                callback(msg)
        finally:
            self.node = old

    def feed(self, x, y, yaw, *, pose=True, imu=True, road=True, rows=2, source_delay=0.):
        stamp = Stamp(self.now-source_delay)
        if pose:
            msg=PoseStamped()
            msg.header.stamp,msg.header.frame_id=stamp,'map'
            msg.pose.position.x,msg.pose.position.y=x,y
            msg.pose.orientation.z,msg.pose.orientation.w=math.sin(yaw/2),math.cos(yaw/2)
            self.publish('/vehicle_pose',msg)
        if imu:
            msg=Imu()
            msg.header.stamp=stamp
            msg.orientation.z=math.degrees(yaw)
            self.publish('/imu_data',msg)
        if road:
            msg=Points()
            msg.header.stamp,msg.header.frame_id=stamp,'map'
            # Fixed cones on a straight 3m course, selecting nearest forward rows.
            first=5.*(math.floor(x/5.)+1)
            for cx in (first,first+5.)[:rows]:
                for cy in (-1.5,1.5):
                    msg.cones.append(types.SimpleNamespace(x=cx,y=cy,confidence=1.))
            self.publish('/cone_map',msg)

    def tick(self, index):
        if index % 2 == 0:
            self.node='straight_line_planner_realtime'
            self.planner.planning_timer(None)
        self.node='path_tracking_controller'
        self.controller.control_loop(None)
        return json.loads(self.latest['/control/legacy_status'].data)


def run_case(name, *, speed=2., initial_y=.6, initial_yaw=0., duration=12.,
             wheelbase=1.6, actuator_lag=.1, actuator_delay=0., steering_gain=1.,
             drop=None, rows=2, parameter_overrides=None):
    h=Harness(parameter_overrides)
    x,y,yaw=0.,initial_y,initial_yaw
    wheel=0.
    history=deque()
    records=[]
    status=None
    # A real waiting stage is exercised before any valid inputs.
    for i in range(10):
        h.now=100.+i*.05
        status=h.tick(i)
        if status['state']!='waiting':
            raise AssertionError('Did not wait before data')
    for i in range(int(duration/.05)):
        elapsed=i*.05
        h.now=101.+elapsed
        active_drop = drop and drop[1] <= elapsed < drop[2]
        unavailable=drop[0] if active_drop else None
        h.feed(x,y,yaw,pose=unavailable!='pose',imu=unavailable!='imu',
               road=unavailable!='road',rows=rows)
        status=h.tick(i)
        command=h.latest.get('/control/legacy_candidate_steering',Scalar(0.)).data or 0.
        history.append((elapsed,command))
        while len(history)>1 and history[1][0] <= elapsed-actuator_delay:
            history.popleft()
        delayed=history[0][1] if history[0][0] <= elapsed-actuator_delay else 0.
        wheel += (steering_gain*delayed-wheel)*min(1.,.05/max(.05,actuator_lag))
        # Assumption: steering code radians equal front-wheel radians times gain.
        v=speed if status['state']=='running' else 0.
        records.append(dict(t=elapsed,x=x,y=y,yaw=yaw,command=command,wheel=wheel,
                            state=status['state'],stop_reason=status['stop_reason'],
                            filtered_lateral=h.samples[-1]['lateral_error'] if h.samples else None,
                            filtered_heading=h.samples[-1]['heading_error'] if h.samples else None,
                            control_records=len(h.samples)))
        x+=v*math.cos(yaw)*.05
        y+=v*math.sin(yaw)*.05
        yaw+=v/wheelbase*math.tan(wheel)*.05
    running=[r for r in records if r['state']=='running']
    states=[]
    for record in records:
        if not states or states[-1]['state'] != record['state']:
            states.append(dict(t=record['t'],state=record['state'],reason=record['stop_reason']))
    summary=dict(name=name, started=bool(running), stop_reason=status['stop_reason'],
                 initial_y=initial_y,final_y=y,final_heading_deg=math.degrees(yaw),
                 max_abs_y=max(abs(r['y']) for r in records),
                 command_count=h.messages.get('/control/legacy_candidate_steering',0),
                 full_control_records=len(h.samples),errors=h.errors,
                 states=states,stop_log_count=len(h.stop_events),
                 assumptions=dict(speed=speed,wheelbase=wheelbase,actuator_lag=actuator_lag,
                                  actuator_delay=actuator_delay,steering_gain=steering_gain),
                 drop=drop, rows=rows)
    summary['offline_parameter_overrides']=parameter_overrides or {}
    deltas=[abs(b['command']-a['command']) for a,b in zip(records,records[1:])
            if a['state']=='running' and b['state']=='running']
    step=h.controller.max_steering_step
    summary['rate_limited_fraction']=sum(v>=step-1e-9 for v in deltas)/max(1.,float(len(deltas)))
    h.controller.shutdown()
    return summary,records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(PACKAGE.parents[1]/'exports'/'closed_loop_20261019'))
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    scenarios=[('stationary',dict(speed=0.)),('offset_right',{}),
               ('offset_left',dict(initial_y=-.6)),
               ('heading_right',dict(initial_y=0.,initial_yaw=math.radians(10))),
               ('heading_left',dict(initial_y=0.,initial_yaw=-math.radians(10))),
               ('offset_and_heading',dict(initial_yaw=math.radians(10))),
               ('offset_and_heading_left',dict(initial_y=-.6,initial_yaw=-math.radians(10))),
               ('one_row',dict(rows=1)),
               ('fast_delayed',dict(speed=4.1667,actuator_lag=.2,actuator_delay=.15)),
               ('short_road_gap',dict(drop=('road',3.,3.2))),
               ('long_road_gap',dict(drop=('road',3.,4.5))),
               ('long_pose_gap',dict(drop=('pose',3.,4.5))),
               ('long_imu_gap',dict(drop=('imu',3.,4.5))),
               ('opposite_actuator_sign',dict(steering_gain=-1.))]
    summaries=[]
    for name,config in scenarios:
        summary,records=run_case(name,**config)
        summaries.append(summary)
        expected_stop = (config.get('drop',(None,0,0))[0]+'_stale'
                         if name.startswith('long_') else None)
        summary['expected_stop'] = expected_stop
        summary['control_execution_ok'] = bool(summary['started'] and not summary['errors'] and
                                               summary['full_control_records'] > 1)
        if name == 'opposite_actuator_sign':
            summary['result'] = 'negative_control_not_a_vehicle_calibration'
        elif expected_stop:
            summary['result'] = ('pass' if summary['stop_reason'] == expected_stop and
                                 records[-1]['state']=='stopped' and records[-1]['command']==0.
                                 else 'failure')
        elif name == 'stationary':
            summary['result'] = 'pass' if summary['stop_reason'] is None else 'failure'
        else:
            converged = (summary['stop_reason'] is None and abs(summary['final_y']) < .1 and
                         abs(summary['final_heading_deg']) < 5. and summary['max_abs_y'] < 1.5)
            summary['result'] = ('pass' if converged else
                                 'robustness_failure_under_assumptions' if name=='fast_delayed' else 'failure')
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    hashes={name:hashlib.sha256((PACKAGE/'src'/name).read_bytes()).hexdigest() for name in
            ('genzong_realtime.py','zhixian_realtime.py','track_frame.py','legacy_run_guard.py')}
    report=dict(scenarios=summaries, source_sha256=hashes, vehicle_parameters_calibrated=False,
                all_scenarios_pass=False if any(s['result'] in ('failure','robustness_failure_under_assumptions')
                                                for s in summaries) else True,
                illustrative_convergence_criteria=dict(final_abs_y_lt_m=.1,
                    final_abs_heading_lt_deg=5.,center_max_abs_y_lt_m=1.5), limitations=[
        'Actual planner and complete control_loop; synchronous in-memory transport, no ROS or CAN.',
        'Synthetic map/pose bypass GPS, mapping, LiDAR association and real source timing.',
        'Wheelbase/gain/lag are assumptions, not measured vehicle calibration.',
        'Simple kinematic plant; simulated stop is immediate, not validated VCU braking.',
        'Center position only; full vehicle envelope and tyre dynamics not modelled.'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for s in summaries:
        print('%s: started=%s final_y=%.3f max_y=%.3f stop=%s records=%s errors=%s result=%s' %
              (s['name'],s['started'],s['final_y'],s['max_abs_y'],s['stop_reason'],
               s['full_control_records'],len(s['errors']),s['result']))
    if any(not s['control_execution_ok'] or s['result']=='failure' for s in summaries):
        return 1
    return 2 if any(s['result']=='robustness_failure_under_assumptions' for s in summaries) else 0


if __name__=='__main__':
    sys.exit(main())
