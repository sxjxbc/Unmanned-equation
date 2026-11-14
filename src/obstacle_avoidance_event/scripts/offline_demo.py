#!/usr/bin/env python
"""Synthetic assumptions only. No ROS imports or network access."""
import os,sys,math,json,argparse
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'))
from obstacle_avoidance_event.model import Scene
from obstacle_avoidance_event.pipeline import Pipeline

VEHICLE = dict(wheelbase=.5,front=.3,rear=.2,width=.3,margin=.05,
               max_steer=.7,steer_rate=10.,speed_cap=.5,lateral_accel=.5,
               brake=.5,latency=.2,stop_margin=.3)

def scene(kind):
    centers = [(float(i), .5*math.sin(i/3.) if kind == 's_curve' else 0.) for i in range(0,16,2)]
    cones = []
    for x,y in centers:
        slope = .5/3.*math.cos(x/3.) if kind == 's_curve' else 0.
        norm = math.sqrt(1+slope*slope)
        for side in (-1,1):
            cones.append((x-side*1.75*slope/norm,y+side*1.75/norm))
    def free(p,r):
        # Synthetic known-free tube around generating centerline.
        target = .5*math.sin(p[0]/3.) if kind == 's_curve' else 0.
        return -.8 < p[0]-r and p[0]+r < 15 and abs(p[1]-target)+r*1.2 < 1.5
    return Scene(cones,(.5,0.,0.),1.,free=free,feedback=(0.,0.,1.))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='offline_results.json')
    args=parser.parse_args()
    results={k:Pipeline(VEHICLE).step(scene(k),1.,True) for k in ('straight','s_curve')}
    with open(args.output,'w') as f:
        json.dump(results,f,indent=2)
    for name,result in results.items():
        sc=scene(name)
        elements=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="350" viewBox="-50 -175 1000 350">']
        elements.append('<text x="0" y="-140">SYNTHETIC ONLY: '+name+'</text>')
        for x,y in sc.cones:
            elements.append('<circle cx="%s" cy="%s" r="5" fill="orange"/>'%(x*55,-y*55))
        pts=' '.join('%s,%s'%(x*55,-y*55) for x,y in result['path'])
        elements.append('<polyline points="'+pts+'" fill="none" stroke="blue" stroke-width="2"/></svg>')
        with open(args.output+'.'+name+'.svg','w') as f:
            f.write(''.join(elements))
    for k,v in results.items():
        print(k,v['state'],v['valid'],v['reason'],len(v['path']))
    return 0 if all(v['valid'] for v in results.values()) else 1
if __name__ == '__main__':
    sys.exit(main())
