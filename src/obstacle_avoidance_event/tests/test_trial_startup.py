import os,sys,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
from obstacle_avoidance_event.planning_bridge import PlanningShadowBridge
from obstacle_avoidance_event.input_adapter import AlignedIngress
from obstacle_avoidance_event.low_speed_output import LowSpeedOutputGuard
from test_input_adapter import message,Object
from test_low_speed_output import Client


VEHICLE=dict(pose_reference='rear_axle',geometry_mode='rear_axle_rectangle',wheelbase=1.55,
    front=2.22918,rear=.70082,width=1.4,margin=.1,max_steer=.5,steer_rate=.24,
    speed_cap=1.,lateral_accel=.5,brake=.5,latency=.3,stop_margin=.5)


def bridge(permission=True):
    return PlanningShadowBridge(AlignedIngress(cones_frame='rear_axle',pose_reference='rear_axle'),
        VEHICLE,steering_sign=-1,boundary_mode='inferred_low_speed_trial',trial_permission=permission,
        ready_frames=10,candidate_hold=.5)


def feed(b,stamp,x=0.,front_only=False):
    pose=message(stamp);pose.pose.position.x=x
    cones=message(stamp,frame='rear_axle')
    cones.poses=[Object(position=Object(x=float(i)-x,y=y)) for i in range(2 if front_only else -4,20,2) for y in (-1.75,1.75)]
    b.add_pose(pose);b.add_cones(cones)
    return b.tick(stamp)


class TrialStartupTests(unittest.TestCase):
    def test_real_size_admission_requires_ten_valid_frames(self):
        b=bridge();results=[feed(b,1.+i*.1) for i in range(12)]
        self.assertFalse(any(r.get('drive_eligible') for r in results[:10]))
        self.assertTrue(results[-1]['drive_eligible'],results[-1])
        self.assertFalse(results[-1]['boundary_certified'])
        self.assertEqual(results[-1]['boundary_source'],'inferred_boundary_low_speed_trial')
        self.assertEqual(results[-1]['execution']['action'],'steer')

    def test_permission_missing_never_admits(self):
        b=bridge(False)
        results=[feed(b,1.+i*.1) for i in range(12)]
        self.assertFalse(any(r.get('drive_eligible') for r in results))
        self.assertFalse(b.runner.execution.started)

    def test_rear_observation_not_replaced_by_flag(self):
        b=bridge()
        out=feed(b,1.,front_only=True)
        self.assertFalse(out.get('drive_eligible',False))
        self.assertEqual(out['reason'],'rear_boundary_support_unavailable')

    def test_ideal_planning_to_distance_stop(self):
        b=bridge();client=Client();guard=LowSpeedOutputGuard(client,enabled=True)
        def dispatch(r,now):
            guard.receive(dict(valid=r.get('candidate_valid',False),drive_eligible=r.get('drive_eligible',False),
                steering=r.get('execution',{}).get('steering'),source_stamp=r.get('source_stamp'),
                speed_limit=r.get('speed_limit'),estimated_speed=r.get('output_speed_estimate'),
                rear_position=r.get('rear_position')),now)
            return guard.tick(now)
        for i in range(12):out=dispatch(feed(b,1.+i*.1),1.+i*.1)
        self.assertEqual(out['action'],'steer')
        for i in range(1,32):
            now=2.1+i*.2
            out=dispatch(feed(b,now,i*.1),now)
            if out['action']=='stop':break
        self.assertEqual(out['stop_reason'],'test_distance_complete')
        self.assertTrue(client.requests[-1][1])


if __name__=='__main__':unittest.main()
