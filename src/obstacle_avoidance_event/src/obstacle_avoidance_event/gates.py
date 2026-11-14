"""Bounded Bowyer-Watson Delaunay, no SciPy runtime dependency."""
import math
from .model import distance

def circle(a, b, c):
    d = 2*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
    if abs(d) < 1e-10:
        return None
    aa, bb, cc = [p[0]*p[0]+p[1]*p[1] for p in (a,b,c)]
    x = (aa*(b[1]-c[1])+bb*(c[1]-a[1])+cc*(a[1]-b[1]))/d
    y = (aa*(c[0]-b[0])+bb*(a[0]-c[0])+cc*(b[0]-a[0]))/d
    return (x,y), distance((x,y),a)**2

def triangulate(points):
    n = len(points)
    if n < 3 or n > 120 or len(set(points)) != n:
        raise ValueError('invalid_point_count_or_duplicates')
    span = max(max(p[j] for p in points)-min(p[j] for p in points) for j in (0,1))
    if span < 1e-6:
        raise ValueError('degenerate_points')
    cx = sum(p[0] for p in points)/n
    cy = sum(p[1] for p in points)/n
    pts = list(points)+[(cx-32*span,cy-16*span),(cx,cy+32*span),(cx+32*span,cy-16*span)]
    triangles = [(n,n+1,n+2)]
    for i in range(n):
        bad = []
        for t in triangles:
            cr = circle(*(pts[k] for k in t))
            if cr and distance(pts[i],cr[0])**2 <= cr[1]+1e-9:
                bad.append(t)
        counts = {}
        for t in bad:
            for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
                e = tuple(sorted((a,b)))
                counts[e] = counts.get(e,0)+1
        triangles = [t for t in triangles if t not in bad]
        triangles.extend((a,b,i) for (a,b),count in sorted(counts.items()) if count == 1)
    triangles = [t for t in triangles if max(t) < n and circle(*(pts[k] for k in t))]
    if not triangles:
        raise ValueError('degenerate_triangulation')
    return triangles

def gates(points, width_min=2.5, width_max=4.5):
    triangles = triangulate(points)
    edges = set()
    for t in triangles:
        edges.update(tuple(sorted(e)) for e in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])))
    candidates = {}
    for e in sorted(edges):
        a,b = [points[i] for i in e]
        w = distance(a,b)
        if width_min <= w <= width_max:
            candidates[e] = {'id':e,'center':((a[0]+b[0])/2.,(a[1]+b[1])/2.), 'width':w, 'endpoints':(a,b)}
    adjacent = dict((e,set()) for e in candidates)
    for t in triangles:
        present = [e for e in candidates if set(e).issubset(t)]
        for a in present:
            adjacent[a].update(b for b in present if b != a)
    return candidates, adjacent
