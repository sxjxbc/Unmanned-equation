#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Real ROS integration checks, always isolated and vehicle-output disabled.

Compatible with target Python 2.7. Run from a sourced Linux validation workspace.
No actual sensors or CAN; the parent launches a dedicated master for each case.
"""
from __future__ import print_function
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time

CASES = ('normal', 'startup_recovery', 'short_road_gap', 'long_road_gap',
         'long_gps_gap', 'long_imu_gap', 'stale_imu_stamp')


def write_json(path, value):
    with open(path, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def probe(case, folder):
    import threading
    import rospy
    from sensor_msgs.msg import NavSatFix, Imu
    from geometry_msgs.msg import PoseArray, Pose, PoseStamped
    from nav_msgs.msg import Path
    from std_msgs.msg import String, Float64
    rospy.init_node('fault_validation_probe', anonymous=True, disable_signals=True)
    lock = threading.RLock()
    observations, candidates, times = [], [], {}
    def status(message):
        with lock:
            observations.append((time.time(), json.loads(message.data)))
    def candidate(message):
        with lock:
            candidates.append((time.time(), message.data))
    def received(message, name):
        with lock:
            times.setdefault(name, []).append(time.time())
    subs = [rospy.Subscriber('/control/legacy_status', String, status),
            rospy.Subscriber('/control/legacy_candidate_steering', Float64, candidate)]
    for topic, kind in (('/vehicle_pose',PoseStamped),('/planned_path',Path),
                        ('/planning/legacy_road_status',String)):
        subs.append(rospy.Subscriber(topic,kind,received,callback_args=topic))
    gps_pub=rospy.Publisher('/GPS_data',NavSatFix,queue_size=10)
    imu_pub=rospy.Publisher('/imu_data',Imu,queue_size=10)
    lidar_pub=rospy.Publisher('/clustered_points',PoseArray,queue_size=10)
    start=time.time()
    ready=None
    recovery_delay=4.0 if case=='startup_recovery' else 0.0
    normal_duration=20.0 if case=='normal' else 7.0
    fault_duration=.2 if case=='short_road_gap' else 1.4
    data=[]
    c,s=math.cos(math.radians(.5)),math.sin(math.radians(.5))
    while not rospy.is_shutdown() and time.time()-start < 45.0:
        now=time.time()
        with lock:
            running=[t for t, item in observations if item.get('state')=='running']
            if ready is None and running:
                ready=running[0]
        if ready is not None and now-ready >= normal_duration:
            break
        elapsed=now-ready if ready is not None else -1.
        fault=ready is not None and case not in ('normal','startup_recovery') and 3.0 <= elapsed < 3.0+fault_duration
        stamp=rospy.Time.now()
        imu=Imu();imu.header.stamp=stamp;imu.orientation.z=0.
        if fault and case=='stale_imu_stamp':
            imu.header.stamp=stamp-rospy.Duration(2.0)
        gps=NavSatFix();gps.header.stamp=stamp
        gps.latitude,gps.longitude=39.9260648,116.1992637
        gps.status.status=0
        cones=PoseArray();cones.header.stamp=stamp;cones.header.frame_id='velodyne'
        if not (fault and case in ('short_road_gap','long_road_gap')):
            for x in (5.,10.):
                for y in (-1.8,1.2):
                    point=Pose();dx,dy=x-1.76618,y+.11304
                    point.position.x=c*dx+s*dy
                    point.position.y=-(-s*dx+c*dy)
                    point.position.z=-.4
                    cones.poses.append(point)
        if not (fault and case=='long_imu_gap'):
            imu_pub.publish(imu)
        if now-start >= recovery_delay and not (fault and case=='long_gps_gap'):
            gps_pub.publish(gps)
        lidar_pub.publish(cones)
        data.append(dict(t=now-start,run_t=elapsed,fault=bool(fault)))
        time.sleep(.1)
    with lock:
        observations=list(observations);candidates=list(candidates)
    checks={}
    checks['reached_running']=ready is not None
    checks['output_always_disabled']=bool(observations) and all(item.get('enabled') is False for _,item in observations)
    topics=[name for name,kind in rospy.get_published_topics('/')]
    checks['vehicle_command_topics_absent']=not any(name in topics for name in ('/control/steering','/fssim/cmd'))
    stops=[(t,item) for t,item in observations if item.get('state')=='stopped']
    expected={'long_road_gap':'road_stale','long_gps_gap':'pose_stale',
              'long_imu_gap':'imu_stale','stale_imu_stamp':'imu_stale'}.get(case)
    if expected:
        checks['expected_stop']=bool(stops) and stops[0][1].get('stop_reason')==expected
        checks['stop_latched_after_recovery']=bool(stops) and all(
            item.get('state')=='stopped' for t,item in observations if t>=stops[0][0])
        checks['candidate_zero_after_stop']=bool(stops) and all(
            abs(value)<1e-12 for t,value in candidates if t>=stops[0][0]+.2)
        checks['post_recovery_observation_exists']=bool(stops) and observations[-1][0] > ready+5.0
    else:
        checks['no_spurious_stop']=not stops
        checks['still_running']=bool(observations) and observations[-1][1].get('state')=='running'
    if case=='startup_recovery':
        checks['waited_before_recovery']=any(t-start<recovery_delay and item.get('state')=='waiting' for t,item in observations)
    baseline=[t for t,value in candidates if ready is not None and ready+1.0<=t<ready+(18.0 if case=='normal' else 2.8)]
    rate=(len(baseline)-1)/(baseline[-1]-baseline[0]) if len(baseline)>1 else 0.
    checks['continuous_control_rate']=rate>=10.
    summary=dict(case=case,pass_all=all(checks.values()),checks=checks,
                 startup_seconds=ready-start if ready else None,
                 candidate_hz=rate,candidate_count=len(candidates),status_count=len(observations),
                 stop_reason=stops[0][1].get('stop_reason') if stops else None,
                 stop_delay_s=stops[0][0]-(ready+3.) if stops and ready else None,
                 topic_receive_counts=dict((name,len(values)) for name,values in times.items()),
                 limitations=['Stationary synthetic input; no real CAN/VCU braking or dynamic driving.'])
    write_json(os.path.join(folder,'result.json'),summary)
    write_json(os.path.join(folder,'observations.json'),dict(status=observations,candidates=candidates,injections=data))
    print(json.dumps(summary,sort_keys=True))
    return 0 if summary['pass_all'] else 1


def terminate(children):
    for child in reversed(children):
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
    deadline=time.time()+5
    while time.time()<deadline and any(child.poll() is None for child in children):
        time.sleep(.1)
    for child in reversed(children):
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL)
        child.wait()


def launch_case(case,folder):
    os.makedirs(folder)
    listener=socket.socket();listener.bind(('127.0.0.1',0));port=listener.getsockname()[1];listener.close()
    env=os.environ.copy()
    env.update(ROS_MASTER_URI='http://127.0.0.1:%d'%port,ROS_HOSTNAME='127.0.0.1',ROS_LOG_DIR=folder)
    env.pop('ROS_IP',None)
    children=[];handles=[]
    try:
        for command,name in ((['roscore','-p',str(port)],'master'),
            (['roslaunch','acceleration_event','validation.launch','use_sim_time:=false',
              'synthetic_inputs:=false','log_dir:='+folder],'nodes')):
            handle=open(os.path.join(folder,name+'.log'),'w');handles.append(handle)
            children.append(subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,preexec_fn=os.setsid))
            time.sleep(2.)
        handle=open(os.path.join(folder,'probe.log'),'w');handles.append(handle)
        child=subprocess.Popen([sys.executable,os.path.abspath(__file__),'--probe',case,'--output',folder],
            env=env,stdout=handle,stderr=subprocess.STDOUT,preexec_fn=os.setsid)
        children.append(child)
        deadline=time.time()+55
        while child.poll() is None and time.time()<deadline:
            time.sleep(.1)
        if child.poll() is None:
            raise RuntimeError('Probe timeout')
        path=os.path.join(folder,'result.json')
        if not os.path.exists(path):
            raise RuntimeError('Missing result; inspect '+folder)
        with open(path) as stream:
            result=json.load(stream)
        if any(p.poll() is not None for p in children[:-1]):
            result['pass_all']=False
            result['process_failure']=True
        return result
    finally:
        terminate(children)
        for handle in handles: handle.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--probe',choices=CASES)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    args=parser.parse_args()
    if args.probe:
        return probe(args.probe,args.output)
    root=os.path.abspath(args.output)
    if not os.path.isdir(root): os.makedirs(root)
    results=[]
    for case in args.cases:
        result=launch_case(case,os.path.join(root,case))
        results.append(result)
        print('%s: %s hz=%.2f stop=%s' % (case,result['pass_all'],result['candidate_hz'],result['stop_reason']))
        sys.stdout.flush()
        write_json(os.path.join(root,'report.json'),dict(pass_all=all(r['pass_all'] for r in results),cases=results))
    return 0 if all(r['pass_all'] for r in results) else 1


if __name__=='__main__':
    sys.exit(main())
