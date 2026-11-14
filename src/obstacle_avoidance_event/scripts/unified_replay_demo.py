#!/usr/bin/env python
"""Real core planner and supervisor, known-map pose replay. No vehicle dynamics."""
import os,sys,math,json,argparse
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.dirname(os.path.abspath(__file__))]
from offline_demo import VEHICLE
from trackdrive_finish_demo import feedback
from obstacle_avoidance_event.model import wrap
from obstacle_avoidance_event.pipeline import Pipeline
from obstacle_avoidance_event.closed_route import ClosedRoute,RouteProgress
from obstacle_avoidance_event.trackdrive import TrackdriveMission,TrackdriveSupervisor
from obstacle_avoidance_event.finish import FinishController
from obstacle_avoidance_event.execution_adapter import LocalExecutionAdapter
from obstacle_avoidance_event.scene_adapter import SceneAssembler

def run(max_steps=4000):
    radius=32.
    route=ClosedRoute([(radius*math.cos(2*math.pi*i/400.),radius*math.sin(2*math.pi*i/400.)) for i in range(400)])
    supervisor=TrackdriveSupervisor(Pipeline(VEHICLE),TrackdriveMission(route.length,3,.5,.2),RouteProgress(route,3),FinishController(1.76618,1.16382,1.4,.5,.2))
    rows=[]
    execution=LocalExecutionAdapter(1,VEHICLE['max_steer'])
    assembler=SceneAssembler(synthetic=True)
    for i in range(max_steps):
        angle=-.08+max(0,i-1)*.005
        now=1. if i==0 else 6.+max(0,i-1)*.8
        speed=0. if i<2 else .2
        pose=(radius*math.cos(angle),radius*math.sin(angle),angle+math.pi/2.)
        cones=[]
        for k in range(128):
            a=2*math.pi*k/128.
            advance=radius*wrap(a-angle)
            if -2.<advance<12.:
                for rad in (radius-1.5,radius+1.5):cones.append((rad*math.cos(a),rad*math.sin(a)))
        aligned=dict(cones=cones,pose=pose,stamp=now,frame='synthetic_map',pose_reference='rear_axle')
        free_space=dict(free=lambda p,r:abs(math.hypot(p[0],p[1])-radius)+r<1.5,
                        stamp=now,frame='synthetic_map',version=1,source='synthetic')
        scene=assembler.build(aligned,free_space,dict(speed=speed,steering=0.,stamp=now,source='synthetic'),now,1)
        # Curved circular finish zone has NOT been certified as a straight stop zone.
        result=supervisor.step(scene,now,signals=feedback(now,speed,go=i==1))
        command=execution.step(result,scene.stamp,now)
        rows.append(dict(time=now,pose=pose,reason=result['reason'],candidate_valid=result['candidate_valid'],
                         laps=result.get('mission',{}).get('laps'),phase=result.get('mission',{}).get('phase'),
                         execution=command))
        if result['reason']:break
    return dict(rows=rows,last=rows[-1],vehicle_output=False,
                recorded_requests=execution.client.requests,
                assumptions='exact known-map pose replay; synthetic small vehicle; not motion controlled by commands')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args();result=run()
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(result['last'])
    expected=result['last']['reason']=='stop_geometry_unconfirmed' and result['last']['laps']==3
    sys.exit(0 if expected else 1)
