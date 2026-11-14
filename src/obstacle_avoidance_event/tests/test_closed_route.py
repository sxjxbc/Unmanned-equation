import os,sys,math,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from obstacle_avoidance_event.closed_route import ClosedRoute,RouteProgress
from closed_route_demo import run

class ClosedRouteTests(unittest.TestCase):
    def tracker(self):
        route=ClosedRoute([(10*math.cos(2*math.pi*i/200.),10*math.sin(2*math.pi*i/200.)) for i in range(200)])
        return RouteProgress(route,3)
    def pose(self,a):return (10*math.cos(a),10*math.sin(a),a+math.pi/2.)
    def test_three_laps_from_pose(self):
        result=run();self.assertEqual(result['phase'],'STOPPING')
        self.assertEqual(result['laps'],3);self.assertEqual(result['gate_events'],4)
    def test_teleport_rejected(self):
        t=self.tracker();t.update(self.pose(0),1,1)
        with self.assertRaises(ValueError):t.update(self.pose(1),1.1,1)
    def test_reverse_rejected(self):
        t=self.tracker();t.update(self.pose(.5),1,1)
        with self.assertRaises(ValueError):t.update(self.pose(.4),2,1)
    def test_version_rejected(self):
        with self.assertRaises(ValueError):self.tracker().update(self.pose(0),1,2)
    def test_time_rejected(self):
        t=self.tracker();t.update(self.pose(0),1,1)
        with self.assertRaises(ValueError):t.update(self.pose(0),1,1)
    def test_gate_jitter_not_rearmed(self):
        t=self.tracker();events=[]
        for i,a in enumerate((-.05,-.02,.001,-.001,.001,-.001,.001)):
            events.append(t.update(self.pose(a),i+1,1)['gate'])
        self.assertEqual(sum(events),1)
    def test_reverse_heading_rejected(self):
        p=self.pose(0)
        with self.assertRaises(ValueError):self.tracker().update((p[0],p[1],p[2]+math.pi),1,1)
    def test_nearby_parallel_segment_not_selected(self):
        route=ClosedRoute([(0,0),(10,0),(10,1),(0,1)])
        t=RouteProgress(route,3,max_error=.6)
        t.update((2,.4,0),1,1)
        # Close return leg has opposite heading and distant arc progress.
        r=t.update((2.1,.6,0),1.1,1)
        self.assertEqual(r['segment'],0)
if __name__=='__main__':unittest.main()
