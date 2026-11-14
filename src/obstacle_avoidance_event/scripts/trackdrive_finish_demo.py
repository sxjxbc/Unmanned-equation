#!/usr/bin/env python
"""Known-map lap replay + scalar finish simulation, candidate only."""
import os,sys,json,argparse
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.dirname(os.path.abspath(__file__))]
from closed_route_demo import run
from obstacle_avoidance_event.finish import StartupGate,FinishController

def feedback(t,speed=0,**overrides):
    d=dict(emergency=False,ready=True,go=False,driving=True,sound_done=True,speed=speed,as_finished=False,ebs_active=True,drive_disabled=True)
    d.update(overrides)
    return dict((k,(v,t)) for k,v in d.items())

def evaluate():
    gate=StartupGate();gate.step(0,feedback(0))
    start=gate.step(5,feedback(5,go=True))
    if not start['permitted']:raise ValueError('startup_not_permitted')
    laps=run()
    if laps['phase']!='STOPPING':raise ValueError('laps_not_completed')
    f=FinishController(1.76618,1.16382,1.4,.5,.2)
    after=0.;speed=2.;rows=[];dt=.05
    for i in range(1000):
        now=400+i*dt
        # Explicit simulated hardware ack after completion request, not real AS.
        ack=bool(rows and rows[-1]['result']['completion_request'])
        result=f.step(now,after,speed,0,feedback(now,speed,as_finished=ack))
        rows.append(dict(time=now,after=after,speed=speed,result=result))
        if result['phase'] in ('FAULT','FINISHED_CONFIRMED'):break
        target=min(2.,result['speed_limit'])
        old=speed;speed=max(0.,speed+max(-.5*dt,min(.3*dt,target-speed)))
        after+=(old+speed)*dt/2.
    return dict(startup=start,laps=laps['laps'],gate_events=laps['gate_events'],
                phase=rows[-1]['result']['phase'],front_after=after+1.76618,
                rows=rows,vehicle_output=False,
                assumptions='exact known-map poses; 1D finish; synthetic brake and hardware acknowledgements')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args();result=evaluate()
    with open(args.output,'w') as f:json.dump(result,f,indent=2)
    print(result['phase'],result['laps'],result['front_after'])
    sys.exit(0 if result['phase']=='FINISHED_CONFIRMED' and result['front_after']<=30 else 1)
