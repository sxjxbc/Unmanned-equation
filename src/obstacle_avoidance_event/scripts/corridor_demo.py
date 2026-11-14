#!/usr/bin/env python
"""Topology-only synthetic scenes; does not authorize vehicle motion."""
import os,sys,math,json,argparse
sys.path[:0]=[os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'),os.path.dirname(os.path.abspath(__file__))]
from obstacle_avoidance_event.gates import gates
from obstacle_avoidance_event.search import search
from obstacle_avoidance_event.model import wrap
from offline_demo import scene

def hairpin():
    points=[]
    for i in range(19):
        t=math.pi*i/18.
        for radius in (3.25,6.75):
            points.append((radius*math.sin(t),5-radius*math.cos(t)))
    return points

def classify(centers):
    headings=[math.atan2(b[1]-a[1],b[0]-a[0]) for a,b in zip(centers,centers[1:])]
    turns=[wrap(b-a) for a,b in zip(headings,headings[1:])]
    cumulative=sum(turns)
    if abs(cumulative)>2.5:return 'left_hairpin' if cumulative>0 else 'right_hairpin'
    if any(t>.01 for t in turns) and any(t<-.01 for t in turns):return 's_curve'
    return 'left' if cumulative>.1 else 'right' if cumulative<-.1 else 'straight'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='corridors.json');args=ap.parse_args()
    cases={'s_curve':scene('s_curve').cones,'left_hairpin':hairpin(),
           'right_hairpin':[(x,-y) for x,y in hairpin()]}
    results={}
    for name,points in cases.items():
        gs,adj=gates(points)
        try:
            centers,ids=search(gs,adj,(0,0,0))
            results[name]=dict(valid=True,classification=classify(centers),centers=centers,gates=ids,vehicle_output=False)
        except ValueError as e:
            results[name]=dict(valid=False,reason=str(e),vehicle_output=False);centers=[];ids=[]
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="800" viewBox="-100 -650 900 800">',
             '<text x="0" y="-600">SYNTHETIC TOPOLOGY ONLY: '+name+'</text>']
        for x,y in points:svg.append('<circle cx="%s" cy="%s" r="4" fill="orange"/>'%(x*50,-y*50))
        for e in ids:
            a,b=[points[i] for i in e]
            svg.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="gray"/>'%(a[0]*50,-a[1]*50,b[0]*50,-b[1]*50))
        svg.append('<polyline points="'+ ' '.join('%s,%s'%(x*50,-y*50) for x,y in centers)+'" fill="none" stroke="blue" stroke-width="2"/></svg>')
        with open(args.output+'.'+name+'.svg','w') as f:f.write(''.join(svg))
    with open(args.output,'w') as f:json.dump(results,f,indent=2)
    for name,result in results.items():print(name,result.get('classification',result.get('reason')))
    return 0 if all(r.get('classification')==n for n,r in results.items()) else 1
if __name__=='__main__':sys.exit(main())
