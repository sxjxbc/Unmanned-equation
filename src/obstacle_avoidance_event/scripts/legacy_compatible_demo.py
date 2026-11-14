#!/usr/bin/env python
"""Local synthetic compatibility check, no hardware feedback or transport."""
import os,sys,json,argparse
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.dirname(os.path.abspath(__file__))]
from offline_demo import scene,VEHICLE
from obstacle_avoidance_event.compatibility import LegacyCompatibleRunner


def run(kind='s_curve'):
    runner=LegacyCompatibleRunner(VEHICLE,1,synthetic=True)
    rows=[]
    for i in range(4):
        stamp=1.+.1*i
        sample=scene(kind)
        aligned=dict(pose=sample.pose,cones=sample.cones,stamp=stamp,
                     frame='synthetic_map',pose_reference='rear_axle')
        free=dict(free=sample.free,stamp=stamp,frame='synthetic_map',version=1,source='synthetic')
        rows.append(runner.step(aligned,free,stamp,stop_requested=i==3))
    return dict(rows=rows,vehicle_output=False,recorded_requests=runner.execution.client.requests,
                scope='synthetic stationary poses; no hardware feedback; not dynamic Trackdrive acceptance')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run()
    with open(args.output,'w') as stream:json.dump(result,stream,indent=2)
    print([row['execution']['action'] for row in result['rows']])
    sys.exit(0 if [r['execution']['action'] for r in result['rows']]==['wait','steer','steer','stop'] else 1)
