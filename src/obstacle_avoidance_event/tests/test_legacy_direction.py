"""Software basis/sign check against original PD, not physical wheel calibration."""
import ast,os,sys,math,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
from obstacle_avoidance_event.control import pursuit
from obstacle_avoidance_event.execution_adapter import LocalExecutionAdapter


class LegacyDirectionTests(unittest.TestCase):
    def test_reflected_target_requires_negative_execution_sign(self):
        filename=os.path.join(ROOT,'..','acceleration_event','src','genzong_realtime.py')
        with open(filename,'rb') as f:tree=ast.parse(f.read(),filename=filename)
        cls=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PathTrackingController'][0]
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='pid_control']
        tree.body=[cls]
        class Logger(object):
            def logdebug_throttle(self,*args):pass
        scope=dict(math=math,rospy=Logger())
        eval(compile(tree,filename,'exec'),scope)
        for side in (-1.,1.):
            old=scope['PathTrackingController']()
            for key,value in dict(kp_lateral=.6,kd_lateral=.05,kp_heading=.2,kd_heading=.01,
                max_lateral_derivative=.6,max_heading_derivative=.35,steering_sign=1.,
                max_steering=.5,max_steering_step=.012,heading_weight_min=.35,
                heading_weight_full_lateral_error=.8,prev_lateral_error=0.,prev_heading_error=0.,prev_steering_angle=0.).items():
                setattr(old,key,value)
            old_angle=old.pid_control(-side*.2,-side*.1,.05)
            new_angle,index=pursuit([(0.,0.),(1.,side*.1),(2.,side*.2)],(0.,0.,0.),
                                    dict(wheelbase=1.55,max_steer=.5))
            out=LocalExecutionAdapter(-1,.5).step(dict(candidate_valid=True,steering=new_angle,speed_limit=.5),1.,1.)
            self.assertGreater(old_angle*out['steering'],0.)
            self.assertLess(old_angle*new_angle,0.)


if __name__=='__main__':unittest.main()
