#!/usr/bin/env python
"""Design-size sensitivity. No vehicle calibration or real localization."""
import os,sys,json,math,argparse,io
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from offline_demo import scene,VEHICLE
from obstacle_avoidance_event.geometry import design_vehicle
from obstacle_avoidance_event.pipeline import Pipeline
from obstacle_avoidance_event.trajectory import validate,spline

def evaluate():
    with io.open(os.path.join(ROOT,'config','design_dimensions.json'),encoding='utf-8') as f:profile=json.load(f)
    results=[]
    for steer in (.35,.5,.7):
        assumptions=dict(VEHICLE);assumptions['max_steer']=steer
        vehicle=design_vehicle(profile,assumptions)
        for kind,obstacles in [('straight',[]),('s_curve',[]),('single',[(5,0,.2)]),('alternating',[(4,.2,.2),(7,-.2,.2)])]:
            obs=scene('s_curve' if kind=='s_curve' else 'straight')
            obs.obstacles=obstacles
            # Explicit synthetic known-free strip, including space behind rear axle.
            obs.free=lambda p,r: -5<p[0]-r and p[0]+r<20 and abs(p[1])+r<1.6
            out=Pipeline(vehicle).step(obs,1.,True)
            results.append(dict(scene=kind,assumed_max_steer=steer,valid=out['valid'],reason=out['reason']))
    turns=[]
    for radius in (3.,5.,8.):
        for steer in (.35,.5,.7):
            vehicle=design_vehicle(profile,dict(VEHICLE,max_steer=steer))
            # Isolated geometry only; does not lift Pipeline's tight-turn guard.
            points=[(radius*math.sin(math.pi*i/24.),radius*(1-math.cos(math.pi*i/24.))) for i in range(25)]
            obs=scene('straight');obs.cones=[];obs.free=lambda p,r:True
            try:validate(spline(points),obs,vehicle);reason=None
            except ValueError as exc:reason=str(exc)
            turns.append(dict(radius=radius,assumed_max_steer=steer,geometry_valid=reason is None,reason=reason))
    return dict(profile=profile,assumptions='steering/braking/rate from synthetic demo, NOT measured; free strip synthetic',cases=results,turn_geometry=turns,vehicle_output=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    result=evaluate()
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    for row in result['cases']:print(row)
    for row in result['turn_geometry']:print(row)
