import os,sys,unittest,math
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from obstacle_avoidance_event.geometry import design_vehicle,body_disks
from offline_demo import VEHICLE

class DesignGeometryTests(unittest.TestCase):
    def vehicle(self):return design_vehicle(dict(length=2.93,width=1.4,wheelbase=1.55,localization_offset=None),VEHICLE)
    def test_unknown_not_split_equally(self):
        v=self.vehicle();self.assertAlmostEqual(v['front'],2.93);self.assertAlmostEqual(v['rear'],1.38)
    def test_rectangle_union_covered(self):
        v=self.vehicle();disks=list(body_disks((0,0),0,v))
        for x in (-1.38,-.4,0,1.55,2.93):
            for y in (-.7,.7):
                self.assertTrue(any(math.hypot(x-c[0],y-c[1])<=r for c,r in disks))
    def test_rotated_union(self):
        disks=list(body_disks((1,2),math.pi/2,self.vehicle()))
        self.assertTrue(any(math.hypot(1-c[0],4.93-c[1])<=r for c,r in disks))
    def test_design_candidates(self):
        from design_size_validation import evaluate
        results=evaluate()['cases']
        for row in results:
            self.assertEqual(row['valid'],row['scene'] in ('straight','s_curve'))
    def test_unknown_area_still_rejected(self):
        from offline_demo import scene
        from obstacle_avoidance_event.pipeline import Pipeline
        obs=scene('s_curve');obs.free=lambda p,r:False
        out=Pipeline(self.vehicle()).step(obs,1,True)
        self.assertFalse(out['valid']);self.assertEqual(out['reason'],'unknown_or_boundary')
    def test_original_synthetic_unchanged(self):
        self.assertEqual(len(list(body_disks((0,0),0,VEHICLE))),1)
if __name__=='__main__':unittest.main()
