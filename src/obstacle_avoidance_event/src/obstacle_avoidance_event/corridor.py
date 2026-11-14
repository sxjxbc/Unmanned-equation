"""Continuous boundary polygon from EXPLICIT ordered boundary inputs.

Not a classifier for uncolored detections. Does not treat undetected space as free.
"""
import math
from .model import distance

def segment_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/max(dx*dx+dy*dy,1e-12)))
    return distance(p,(a[0]+t*dx,a[1]+t*dy))

def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

class Corridor(object):
    def __init__(self,left,right,confirmed=False,max_gap=5.):
        if not confirmed:raise ValueError('boundary_identity_unconfirmed')
        if min(len(left),len(right))<2:raise ValueError('insufficient_boundary')
        for side in (left,right):
            if any(distance(a,b)>max_gap for a,b in zip(side,side[1:])):raise ValueError('boundary_gap')
        if any(math.isnan(v) or math.isinf(v) for p in list(left)+list(right) for v in p):raise ValueError('nonfinite_boundary')
        self.polygon=list(left)+list(reversed(right))
        self.edges=list(zip(self.polygon,self.polygon[1:]+self.polygon[:1]))
        if any(distance(a,b)<1e-8 for a,b in self.edges):raise ValueError('degenerate_boundary')
        for i,(a,b) in enumerate(self.edges):
            for j,(c,d) in enumerate(self.edges):
                if j<=i+1 or (i==0 and j==len(self.edges)-1):continue
                if orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0:
                    raise ValueError('crossed_boundary')
    def contains(self,p):
        inside=False
        for a,b in self.edges:
            if (a[1]>p[1]) != (b[1]>p[1]):
                x=a[0]+(p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
                if p[0]<x:inside=not inside
        return inside
    def free(self,p,radius):
        return not math.isnan(radius) and not math.isinf(radius) and radius>=0 and self.contains(p) and all(segment_distance(p,a,b)>radius for a,b in self.edges)
