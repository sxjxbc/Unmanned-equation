#!/usr/bin/env python
"""Offline synthetic closed loop. No ROS or network imports."""
import os,sys,math,json,argparse
sys.path[:0]=[os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'),os.path.dirname(os.path.abspath(__file__))]
from offline_demo import scene,VEHICLE
from obstacle_avoidance_event.pipeline import Pipeline
from obstacle_avoidance_event.simulation import Bicycle, audit_motion

def run(kind, stale_at=None, duration=35., delay_steps=0, missing_at=None, obstacles=None, goal_x=8.):
    vehicle=dict(VEHICLE)
    plant=Bicycle(vehicle)
    planner=Pipeline(vehicle)
    dt=.05
    rows=[];reason=None;goal=False
    commands=[(0.,0.)]*delay_steps
    motion_safe=True
    for i in range(int(duration/dt)):
        now=1+i*dt
        obs=scene(kind);obs.pose=plant.pose;obs.stamp=now
        obs.obstacles=list(obstacles or [])
        obs.feedback=(plant.speed,plant.steering,now)
        if stale_at is not None and i>=stale_at:obs.stamp=now-1.
        if missing_at is not None and i>=missing_at:obs.cones=[]
        result=planner.step(obs,now,True)
        if not result['valid'] and reason is None:reason=result['reason']
        if plant.pose[0]>=goal_x:goal=True
        stopping=reason is not None or goal
        speed=0. if stopping else min(.2,result['speed_limit']*.7)
        steering=plant.steering if stopping else result['steering']
        commands.append((steering,speed))
        delayed_steer,delayed_speed=commands.pop(0)
        # Stop arbitration bypasses the experimental command queue.
        if stopping:delayed_speed=0.;delayed_steer=plant.steering
        before=plant.pose;old_speed=plant.speed
        plant.step(delayed_steer,delayed_speed,dt)
        audit=audit_motion(before,plant.pose,obs,vehicle,(old_speed+plant.speed)*dt/2.)
        if audit is not None:
            motion_safe=False
            if reason is None:reason=audit
        x,y,yaw=plant.pose
        reference=.5*math.sin(x/3.) if kind=='s_curve' else 0.
        rows.append(dict(time=now,x=x,y=y,yaw=yaw,speed=plant.speed,
                         steering=plant.steering,error=y-reference,
                         state=result['state'],reason=result['reason'],valid=result['valid'],
                         selected_offset=result.get('selected_offset'),maneuver_released=result.get('maneuver_released',False),
                         vehicle_output=False))
        if stopping and plant.speed<1e-6:break
    return dict(kind=kind,goal_reached=goal,reason=reason,
                stopped=plant.speed<1e-6,motion_safe=motion_safe,delay_steps=delay_steps,max_error=max(abs(r['error']) for r in rows),
                distance=plant.pose[0]-.5,rows=rows,
                assumptions='synthetic small vehicle; exact pose, full map, known free space; no latency/slip')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='closed_loop.json');args=ap.parse_args()
    results={k:run(k,duration=70.) for k in ('straight','s_curve')}
    results['stale_stop']=run('straight',stale_at=30)
    with open(args.output,'w') as f:json.dump(results,f,indent=2)
    for name,result in results.items():
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="350" viewBox="-50 -175 1000 350">',
             '<text x="0" y="-140">SYNTHETIC CLOSED LOOP: '+name+'</text>']
        for x,y in scene(result['kind']).cones:svg.append('<circle cx="%s" cy="%s" r="4" fill="orange"/>'%(x*55,-y*55))
        pts=' '.join('%s,%s'%(r['x']*55,-r['y']*55) for r in result['rows'])
        svg.append('<polyline points="'+pts+'" fill="none" stroke="blue" stroke-width="2"/></svg>')
        with open(args.output+'.'+name+'.svg','w') as f:f.write(''.join(svg))
        print(name,'goal',result['goal_reached'],'reason',result['reason'],'distance',round(result['distance'],3),'max_error',round(result['max_error'],3),'stopped',result['stopped'])
    return 0 if all(results[k]['goal_reached'] and results[k]['reason'] is None and results[k]['stopped'] for k in ('straight','s_curve')) and results['stale_stop']['reason']=='observation_stale' and results['stale_stop']['stopped'] else 1
if __name__=='__main__':sys.exit(main())
