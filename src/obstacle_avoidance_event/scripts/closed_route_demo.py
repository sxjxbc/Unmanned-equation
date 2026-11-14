#!/usr/bin/env python
"""Three-lap known-map localization replay, not a vehicle dynamics simulation."""
import os,sys,math,json,argparse
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
from obstacle_avoidance_event.closed_route import ClosedRoute,RouteProgress
from obstacle_avoidance_event.trackdrive import TrackdriveMission

def run():
    radius=32.
    route=ClosedRoute([(radius*math.cos(2*math.pi*i/400.),radius*math.sin(2*math.pi*i/400.)) for i in range(400)])
    tracker=RouteProgress(route,3.)
    mission=TrackdriveMission(route.length,3.,.5,.2)
    start=-.08
    initial=(route.length+radius*start)%route.length
    mission.step(0,initial,0,ready=True);mission.step(5,initial,0,ready=True,go=True)
    rows=[]
    for i in range(5000):
        angle=start+i*.005
        pose=(radius*math.cos(angle),radius*math.sin(angle),angle+math.pi/2.)
        observation=tracker.update(pose,6+i*.08,1)
        result=mission.step(6+i*.08,observation['progress'],2.,gate=observation['gate'],ready=True,driving_ack=True,sound_done=True)
        rows.append(dict(pose=pose,match=observation,mission=result))
        if result['phase'] in ('STOPPING','FAULT'):break
    return dict(length=route.length,laps=result['laps'],phase=result['phase'],reason=result['reason'],gate_events=sum(row['match']['gate'] for row in rows),rows=rows,vehicle_output=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args();result=run()
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(result['phase'],result['laps'],result['gate_events'],result['reason'])
    sys.exit(0 if result['phase']=='STOPPING' and result['laps']==3 and result['gate_events']==4 else 1)
