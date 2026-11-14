"""Infer a SHADOW-ONLY corridor from uncolored cone gates, not certified space."""
import math
from .gates import gates
from .search import search,oriented_gate,transition_ok
from .model import distance


def infer_boundary(aligned,version=1,previous=None,rear_clearance=0.):
    points=sorted(aligned['cones'])
    graph,adjacency=gates(points)
    centers,sequence=search(graph,adjacency,aligned['pose'],previous)
    heading=math.atan2(centers[1][1]-centers[0][1],centers[1][0]-centers[0][0])
    x,y,yaw=aligned['pose']
    behind=[]
    for key,gate in graph.items():
        if key in sequence:continue
        dx,dy=gate['center'][0]-x,gate['center'][1]-y
        forward=math.cos(yaw)*dx+math.sin(yaw)*dy
        lateral=-math.sin(yaw)*dx+math.cos(yaw)*dy
        if -4.<forward<=-rear_clearance and abs(lateral)<.6 and distance(gate['center'],centers[0])<=5.:
            a,b=gate['endpoints']
            if rear_clearance>0 and abs(math.cos(heading)*(a[0]-b[0])+math.sin(heading)*(a[1]-b[1]))>.5:
                continue # A skewed end cap must not cut through the long body's rear corner.
            if oriented_gate(gate,heading) is not None and transition_ok(gate,graph[sequence[0]],heading,{}):
                behind.append((distance(gate['center'],(x,y)),key))
    if not behind:raise ValueError('rear_boundary_support_unavailable')
    sequence=[min(behind)[1]]+sequence
    left,right=[],[]
    for i,key in enumerate(sequence):
        if i:
            a,b=graph[sequence[i-1]]['center'],graph[key]['center']
            heading=math.atan2(b[1]-a[1],b[0]-a[0])
        sides=oriented_gate(graph[key],heading)
        if sides is None:raise ValueError('inferred_boundary_orientation')
        for side,point in zip((left,right),sides):
            if not side or point!=side[-1]:side.append(point)
    if set(left)&set(right):raise ValueError('inferred_boundary_side_conflict')
    return dict(stamp=aligned['stamp'],frame=aligned['frame'],version=version,confirmed=True,
                source='inferred_boundary_shadow_only',left=left,right=right,
                obstacles=aligned.get('obstacles',[]))
