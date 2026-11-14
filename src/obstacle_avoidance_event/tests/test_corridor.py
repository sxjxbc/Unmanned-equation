import os,sys,unittest,math
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from corridor_demo import hairpin,classify
from obstacle_avoidance_event.gates import gates
from obstacle_avoidance_event.search import search,oriented_gate,transition_ok
from obstacle_avoidance_event.association import ConeTracker

class CorridorTests(unittest.TestCase):
    def test_left_hairpin_order(self):
        pts=hairpin();g,a=gates(pts);c,ids=search(g,a,(0,0,0))
        angles=[math.atan2(x,5-y) for x,y in c]
        self.assertTrue(all(b>a for a,b in zip(angles,angles[1:])))
        self.assertEqual(classify(c),'left_hairpin')
        self.assertGreater(c[-1][1],9)
    def test_right_hairpin(self):
        g,a=gates([(x,-y) for x,y in hairpin()]);c,ids=search(g,a,(0,0,0))
        self.assertEqual(classify(c),'right_hairpin')
    def test_input_order(self):
        pts=list(reversed(hairpin()));g,a=gates(pts);c,ids=search(g,a,(0,0,0))
        self.assertEqual(classify(c),'left_hairpin')
    def test_longitudinal_false_gate(self):
        g={'endpoints':((0,0),(3,0)),'width':3}
        self.assertIsNone(oriented_gate(g,0))
    def test_side_identity_conflict(self):
        g={'endpoints':((0,1.5),(0,-1.5)),'width':3}
        self.assertFalse(transition_ok(g,g,0,{(0,1.5):'right'}))
    def test_symmetric_branch_rejected(self):
        def gate(x,y):return {'center':(x,y),'width':3.,'endpoints':((x,y+1.5),(x,y-1.5))}
        gs={0:gate(1,0),1:gate(2,.3),2:gate(2,-.3),3:gate(3,.6),4:gate(3,-.6)}
        adj={0:set([1,2]),1:set([3]),2:set([4]),3:set(),4:set()}
        with self.assertRaises(ValueError):search(gs,adj,(0,0,0))
    def test_stable_identity_reordered(self):
        t=ConeTracker();first=t.update([(0,0),(3,0)],1,1)
        second=t.update([(3.02,0),(.02,0)],2,1)
        self.assertEqual(first.keys(),second.keys());self.assertEqual(second[0],(.02,0))
    def test_identity_ambiguity_atomic(self):
        t=ConeTracker(radius=1);t.update([(0,0),(.8,0)],1,1)
        before=dict(t.points)
        with self.assertRaises(ValueError):t.update([(.4,0)],2,1)
        self.assertEqual(t.points,before)
    def test_double_match_rejected(self):
        t=ConeTracker();t.update([(0,0)],1,1)
        with self.assertRaises(ValueError):t.update([(.01,0),(.02,0)],2,1)

if __name__=='__main__':unittest.main()
