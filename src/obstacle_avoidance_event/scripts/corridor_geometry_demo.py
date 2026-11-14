#!/usr/bin/env python
"""IMU footprint + continuous synthetic 3m corridor, geometry only."""
import os,sys,math,json,io,argparse
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
from obstacle_avoidance_event.geometry import imu_body_vehicle,body_disks
from obstacle_avoidance_event.corridor import Corridor

def boundaries(kind):
    if kind=='hairpin':
        centers=[(3*math.sin(math.pi*i/40.),3*(1-math.cos(math.pi*i/40.))) for i in range(41)]
    elif kind=='decreasing_radius':
        centers=[];x=y=yaw=0.
        for i in range(61):
            centers.append((x,y));yaw+=.25/(8.-5.*i/60.)
            x+=.25*math.cos(yaw);y+=.25*math.sin(yaw)
    else:centers=[(i*.4,.6*math.sin(i*.4/3.)) for i in range(61)]
    left=[];right=[];headings=[]
    for i,p in enumerate(centers):
        a,b=centers[max(0,i-1)],centers[min(len(centers)-1,i+1)]
        yaw=math.atan2(b[1]-a[1],b[0]-a[0]);headings.append(yaw)
        left.append((p[0]-1.5*math.sin(yaw),p[1]+1.5*math.cos(yaw)))
        right.append((p[0]+1.5*math.sin(yaw),p[1]-1.5*math.cos(yaw)))
    return centers,headings,left,right

def evaluate():
    with io.open(os.path.join(ROOT,'config/design_dimensions.json'),encoding='utf-8') as f:profile=json.load(f)
    vehicle=imu_body_vehicle(profile,.05)
    rows=[]
    for kind in ('s_curve','hairpin','decreasing_radius'):
        points,headings,left,right=boundaries(kind)
        for variant in ('normal','staggered','missing','spare'):
            l,r=list(left),list(right)
            if variant=='staggered':r=r[::2]
            if variant=='missing':l=l[:10]+l[11:]
            # Spare cones deliberately kept as unassigned observations, not boundary.
            spare=[(50.,50.)] if variant=='spare' else []
            corridor=Corridor(l,r,confirmed=True)
            tested=range(8,len(points)-8)
            fits=[all(corridor.free(c,rad) for c,rad in body_disks(points[i],headings[i],vehicle)) for i in tested]
            rows.append(dict(scene=kind,variant=variant,poses=len(fits),fitting_poses=sum(fits),all_fit=all(fits),unassigned_spares=len(spare),boundary_identity='provided_by_synthetic_generator'))
    return rows
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    rows=evaluate()
    with open(args.output,'w') as f:json.dump(rows,f,indent=2)
    for row in rows:print(row)
