import math
from .model import distance, wrap
from .geometry import body_disks

def spline(points, spacing=.15):
    """Parametric cubic Hermite interpolation, sampled in approximate arc length."""
    if len(points) < 3:
        raise ValueError('short_path')
    out = []
    for i in range(len(points)-1):
        a,b = points[i:i+2]
        prev = points[max(0,i-1)]
        nxt = points[min(len(points)-1,i+2)]
        ta = tuple((b[j]-prev[j])/(1. if i == 0 else 2.) for j in (0,1))
        tb = tuple((nxt[j]-a[j])/(1. if i+2 >= len(points) else 2.) for j in (0,1))
        count = max(2,int(distance(a,b)/spacing)+1)
        for k in range(count):
            t = float(k)/count
            p = tuple((2*t**3-3*t*t+1)*a[j]+(t**3-2*t*t+t)*ta[j]+(-2*t**3+3*t*t)*b[j]+(t**3-t*t)*tb[j] for j in (0,1))
            out.append(p)
    out.append(points[-1])
    return out

def curvature(path):
    result = [0.]
    for a,b,c in zip(path,path[1:],path[2:]):
        ab,bc,ac = distance(a,b),distance(b,c),distance(a,c)
        if min(ab,bc,ac) < 1e-8:
            raise ValueError('degenerate_path')
        cross = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        result.append(2*cross/(ab*bc*ac))
    result.append(0.)
    return result

def validate(path, scene, vehicle):
    if scene.free is None:
        raise ValueError('free_space_unavailable')
    # Circumscribed disk is conservative for any body orientation.
    radius = math.hypot(max(vehicle['front'],vehicle['rear']),vehicle['width']/2.)+vehicle['margin']
    ks = curvature(path)
    for i,p in enumerate(path):
        if abs(ks[i]) > math.tan(vehicle['max_steer'])/vehicle['wheelbase']:
            raise ValueError('curvature_limit')
        # Extra half-segment bounds continuous translation between samples.
        pad = max(distance(p,path[max(0,i-1)]),distance(p,path[min(len(path)-1,i+1)]))/2.
        a,b=path[max(0,i-1)],path[min(len(path)-1,i+1)]
        heading=math.atan2(b[1]-a[1],b[0]-a[0])
        # Bound body rotation between adjacent samples for the long union body.
        rotation_pad=0.
        if vehicle.get('geometry_mode') in ('unknown_overhang_union','rear_axle_rectangle'):
            rotation_pad=max(vehicle['front'],vehicle['rear'])*min(math.pi,abs(ks[i])*pad*2.)
        for center,bound in body_disks(p,heading,vehicle,pad+rotation_pad):
            if not scene.free(center,bound):
                raise ValueError('unknown_or_boundary')
            for x,y,r in scene.obstacles+[(x,y,.15) for x,y in scene.cones]:
                if distance(center,(x,y)) <= bound+r:
                    raise ValueError('collision')
    return ks


def select_path(path, scene, vehicle):
    """Finite lateral candidates, always validated against certified free space.

    Not a topology planner. Unknown/boundary failures are not bypassed.
    """
    try:
        return path,validate(path,scene,vehicle),0.
    except ValueError as exc:
        original_reason = str(exc)
        if vehicle.get('geometry_mode') == 'unknown_overhang_union':
            # Bounded chord regularization. Occupancy/footprint tests remain
            # authoritative; no relaxation of unknown space or body bounds.
            cumulative=[0.]
            for a,b in zip(path,path[1:]):cumulative.append(cumulative[-1]+distance(a,b))
            for weight in (.25,.5,.75,1.):
                trial=[]
                for i,p in enumerate(path):
                    t=cumulative[i]/max(cumulative[-1],1e-9)
                    chord=tuple(path[0][j]+t*(path[-1][j]-path[0][j]) for j in (0,1))
                    trial.append(tuple((1-weight)*p[j]+weight*chord[j] for j in (0,1)))
                if max(distance(a,b) for a,b in zip(path,trial)) > .6:
                    continue
                try:
                    return trial,validate(trial,scene,vehicle),0.
                except ValueError:
                    continue
        if original_reason != 'collision':
            raise ValueError(original_reason)
    lengths=[0.]
    for a,b in zip(path,path[1:]):lengths.append(lengths[-1]+distance(a,b))
    total=lengths[-1]
    obstacle_progress = [lengths[min(range(len(path)),key=lambda i:distance(path[i],(x,y)))]
                         for x,y,r in scene.obstacles]
    return_start = max(obstacle_progress) if obstacle_progress else total
    for offset in (.45,-.45,.8,-.8):
        candidate=[]
        for i,p in enumerate(path):
            a,b=path[max(0,i-1)],path[min(len(path)-1,i+1)]
            heading=math.atan2(b[1]-a[1],b[0]-a[0])
            ramp=min(1.,lengths[i]/2.,(total-lengths[i])/2.,max(0.,(return_start+3.-lengths[i])/3.))
            blend=.5-.5*math.cos(math.pi*max(0.,ramp))
            d=offset*blend
            candidate.append((p[0]-math.sin(heading)*d,p[1]+math.cos(heading)*d))
        try:
            return candidate,validate(candidate,scene,vehicle),offset
        except ValueError:
            continue
    # Bounded multi-stage lateral search, at most three distinct obstacles.
    import itertools
    stations=sorted(set(round(v,6) for v in obstacle_progress if .5 < v < total-.5))
    if 0 < len(stations) <= 3:
        for choices in itertools.product((.9,-.9),repeat=len(stations)):
            anchors=[(0.,0.)]+list(zip(stations,choices))+[(min(total,stations[-1]+3.),0.)]
            if anchors[-1][0] <= anchors[-2][0]:
                continue
            candidate=[]
            for i,p in enumerate(path):
                progress=lengths[i]
                d=0.
                for (sa,da),(sb,db) in zip(anchors,anchors[1:]):
                    if sa <= progress <= sb:
                        t=(progress-sa)/(sb-sa)
                        blend=t*t*t*(10.-15.*t+6.*t*t)
                        d=da+(db-da)*blend
                        break
                a,b=path[max(0,i-1)],path[min(len(path)-1,i+1)]
                heading=math.atan2(b[1]-a[1],b[0]-a[0])
                candidate.append((p[0]-math.sin(heading)*d,p[1]+math.cos(heading)*d))
            try:
                return candidate,validate(candidate,scene,vehicle),choices[0]
            except ValueError:
                continue
    raise ValueError('collision')

