"""Known-map closed progress and directed start gate; no perception inference."""
import math
from .model import distance,wrap

class ClosedRoute(object):
    def __init__(self,points,version=1):
        if len(points)<4 or any(math.isnan(v) or math.isinf(v) for p in points for v in p):raise ValueError('invalid_closed_route')
        if distance(points[0],points[-1])<1e-8:points=points[:-1]
        self.points=list(points);self.version=version;self.segments=[];self.length=0.
        for a,b in zip(self.points,self.points[1:]+self.points[:1]):
            length=distance(a,b)
            if length<1e-6:raise ValueError('degenerate_route_segment')
            self.segments.append((a,b,length,self.length));self.length+=length
    def candidates(self,pose):
        x,y,yaw=pose
        candidates=[]
        for i,(a,b,length,start) in enumerate(self.segments):
            dx,dy=b[0]-a[0],b[1]-a[1]
            t=max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/(length*length)))
            point=(a[0]+t*dx,a[1]+t*dy)
            if abs(wrap(math.atan2(dy,dx)-yaw))>.7:continue
            candidates.append((distance((x,y),point),(start+t*length)%self.length,i))
        return candidates

class RouteProgress(object):
    def __init__(self,route,max_speed,max_error=.5,gate_half_width=1.5):
        if max_speed<=0 or max_error<=0 or gate_half_width<=0:raise ValueError('invalid_progress_limits')
        self.route=route;self.max_speed=max_speed;self.max_error=max_error
        self.gate_half_width=gate_half_width;self.last=None;self.unwrapped=None
        self.last_pose=None;self.armed=False;self.cross_distance=None
        a,b=route.points[-1],route.points[1]
        length=distance(a,b);self.direction=((b[0]-a[0])/length,(b[1]-a[1])/length)
    def update(self,pose,stamp,version):
        if version!=self.route.version:raise ValueError('route_version_changed')
        if not all(not math.isnan(v) and not math.isinf(v) for v in tuple(pose)+(stamp,)):raise ValueError('invalid_pose')
        candidates=self.route.candidates(pose)
        choices=[]
        for error,s,index in candidates:
            if error>self.max_error:continue
            if self.last is None:delta=0.
            else:
                dt=stamp-self.last[0]
                if dt<=0:raise ValueError('non_monotonic_pose_time')
                delta=(s-self.last[1]+self.route.length/2.)%self.route.length-self.route.length/2.
                if delta < -.05 or delta > self.max_speed*dt+.2:continue
            choices.append((error,s,index,delta))
        if not choices:raise ValueError('route_match_lost_or_progress_jump')
        choices.sort()
        best=choices[0]
        for alt in choices[1:]:
            separation=abs((alt[1]-best[1]+self.route.length/2.)%self.route.length-self.route.length/2.)
            if alt[0]-best[0]<.02 and separation>1.:raise ValueError('ambiguous_route_match')
        error,s,index,delta=best
        new_progress=s if self.unwrapped is None else self.unwrapped+max(0.,delta)
        origin=self.route.points[0];tx,ty=self.direction
        def coords(p):
            dx,dy=p[0]-origin[0],p[1]-origin[1]
            return dx*tx+dy*ty,-dx*ty+dy*tx
        signed,lateral=coords(pose)
        armed=self.armed
        if signed<-.3 and (self.cross_distance is None or new_progress-self.cross_distance>.8*self.route.length):armed=True
        gate=False
        if self.last_pose is not None:
            old,old_lateral=coords(self.last_pose)
            if armed and old<0<=signed:
                fraction=-old/max(signed-old,1e-12)
                intersection=old_lateral+fraction*(lateral-old_lateral)
                gate=abs(intersection)<=self.gate_half_width
        if gate:armed=False;self.cross_distance=new_progress
        self.armed=armed;self.unwrapped=new_progress
        self.last=(stamp,s);self.last_pose=pose
        return dict(progress=s,unwrapped_progress=new_progress,gate=gate,segment=index,error=error,route_version=version,vehicle_output=False)
