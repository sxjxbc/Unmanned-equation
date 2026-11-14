import ast,json,math,os,sys,threading,unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[os.path.join(ROOT,'src'),os.path.join(ROOT,'scripts')]
from obstacle_avoidance_event.planning_bridge import PlanningShadowBridge,configured_ingress
from obstacle_avoidance_event.input_adapter import AlignedIngress
from offline_demo import VEHICLE,scene
from test_input_adapter import Object,message
from test_shadow_ingress import Publisher,String,Path


def pose_message(stamp):
    m=message(stamp);m.pose.position.x=.5
    return m


def cone_message(stamp):
    m=message(stamp,frame='rear_axle')
    m.poses=[Object(position=Object(x=x-.5,y=y)) for x,y in scene('straight').cones]
    return m


def boundary(stamp):
    return dict(stamp=stamp,frame='map',version=1,confirmed=True,source='synthetic',
                left=[[x,1.5] for x in (-1.,3.,7.,11.,15.)],
                right=[[x,-1.5] for x in (-1.,3.,7.,11.,15.)])


def bridge():
    v=dict(VEHICLE,pose_reference='rear_axle')
    return PlanningShadowBridge(AlignedIngress(cones_frame='rear_axle',pose_reference='rear_axle'),
                                v,synthetic=True,steering_sign=1)


def feed(b,stamp):
    b.add_pose(pose_message(stamp));b.add_cones(cone_message(stamp));b.add_boundary(boundary(stamp))
    return b.tick(stamp)


class BridgeTests(unittest.TestCase):
    def test_inferred_straight_s_curve_and_mirror_multiple_frames(self):
        for kind in ('straight','s_curve'):
            for sign in (-1,1):
                b=PlanningShadowBridge(AlignedIngress(cones_frame='rear_axle',pose_reference='rear_axle'),
                    dict(VEHICLE,pose_reference='rear_axle'),synthetic=True,steering_sign=1,boundary_mode='inferred_shadow')
                for stamp in (1.,1.1,1.2,1.3):
                    cones=cone_message(stamp)
                    cones.poses=[Object(position=Object(x=x-.5,y=sign*y)) for x,y in scene(kind).cones]
                    b.add_pose(pose_message(stamp));b.add_cones(cones)
                    out=b.tick(stamp)
                self.assertTrue(out['candidate_valid'],(kind,sign,out['reason']))

    def test_inferred_shadow_boundary_without_external_packet(self):
        b=PlanningShadowBridge(AlignedIngress(cones_frame='rear_axle',pose_reference='rear_axle'),
            dict(VEHICLE,pose_reference='rear_axle'),synthetic=True,steering_sign=1,boundary_mode='inferred_shadow')
        for stamp in (1.,1.1):
            b.add_pose(pose_message(stamp));b.add_cones(cone_message(stamp))
            out=b.tick(stamp)
        self.assertTrue(out['candidate_valid'],out['reason'])
        self.assertEqual(out['boundary_source'],'inferred_boundary_shadow_only')
        self.assertFalse(out['drive_eligible']);self.assertFalse(out['vehicle_output'])

    def test_real_core_path_without_vehicle_io(self):
        b=bridge();self.assertTrue(feed(b,1.)['waiting'])
        out=feed(b,1.1)
        self.assertTrue(out['candidate_valid'],out['reason'])
        self.assertGreater(len(out['planning']['path']),20)
        self.assertFalse(out['execution']['vehicle_output'])

    def test_repeated_timer_does_not_reestimate_or_rerecord(self):
        b=bridge();feed(b,1.);feed(b,1.1)
        count=len(b.runner.execution.client.requests)
        out=b.tick(1.15)
        self.assertTrue(out['held']);self.assertFalse(out['execution']['request_recorded'])
        self.assertEqual(len(b.runner.execution.client.requests),count)

    def test_stale_hold_latches_and_cannot_revive(self):
        b=bridge();feed(b,1.);feed(b,1.1)
        self.assertEqual(b.tick(1.4)['execution']['action'],'stop')
        self.assertEqual(feed(b,1.5)['execution']['action'],'stop')
        self.assertEqual(b.tick(1.51)['execution']['action'],'stop')

    def test_unconfirmed_or_wrong_boundary_stops(self):
        for field,value in (('confirmed',False),('frame','old_map'),('source','unobserved')):
            b=bridge();feed(b,1.);feed(b,1.1)
            packet=boundary(1.2);packet[field]=value;b.add_boundary(packet)
            self.assertEqual(b.tick(1.2)['execution']['action'],'stop')

    def test_missing_mismatched_and_gapped_boundary_waits(self):
        b=bridge();b.add_pose(pose_message(1.));b.add_cones(cone_message(1.))
        self.assertTrue(b.tick(1.)['waiting'])
        b.add_boundary(boundary(1.1));self.assertFalse(b.tick(1.1)['candidate_valid'])
        packet=boundary(1.2);packet['left']=[[0,1.5],[10,1.5]]
        b.add_boundary(packet);self.assertFalse(b.tick(1.2)['candidate_valid'])

    def test_explicit_obstacle_is_checked(self):
        b=bridge();feed(b,1.)
        b.add_pose(pose_message(1.1));b.add_cones(cone_message(1.1))
        packet=boundary(1.1);packet['obstacles']=[[3.,0.,1.]];b.add_boundary(packet)
        out=b.tick(1.1)
        self.assertFalse(out['candidate_valid']);self.assertEqual(out['execution']['action'],'stop')

    def test_bad_pose_and_map_version_latch(self):
        b=bridge();feed(b,1.);feed(b,1.1)
        b.add_pose(message(1.2,frame='wrong'))
        self.assertEqual(b.tick(1.2)['execution']['action'],'stop')
        b=bridge();feed(b,1.);feed(b,1.1)
        b.add_pose(pose_message(1.2));b.add_cones(cone_message(1.2))
        packet=boundary(1.2);packet['version']=2;b.add_boundary(packet)
        self.assertEqual(b.tick(1.2)['execution']['action'],'stop')

    def test_default_configuration_waits(self):
        self.assertTrue(PlanningShadowBridge().tick(1.)['waiting'])
        with self.assertRaises(ValueError):configured_ingress({})
        cfg=dict(localization_reference='confirmed_imu',convention_confirmed=True,
                 lidar_offset_x=1.,lidar_offset_y=0.,lidar_y_sign=-1,lidar_yaw_offset_deg=0.)
        with self.assertRaises(ValueError):configured_ingress(cfg)

    def test_legacy_messages_pass_through_conversion(self):
        cfg=dict(localization_reference='confirmed_imu',convention_confirmed=True,rear_in_imu=[-.5,0.],
                 lidar_offset_x=1.,lidar_offset_y=0.,lidar_y_sign=-1,lidar_yaw_offset_deg=0.)
        ingress=configured_ingress(cfg)
        m=message(1.);m.pose.position.x=2.;ingress.add_pose(m)
        m=message(1.,frame='/velodyne')
        out=ingress.align(m,1.)
        self.assertAlmostEqual(out['pose'][0],1.5)
        self.assertAlmostEqual(out['cones'][0][0],4.)
        self.assertEqual(out['frame'],'trackdrive_map')


class RosDoubles(object):
    def __init__(self):self.now=1.;self.subscribers=[];self.publishers=[]
    def get_param(self,key,default):return default
    def Publisher(self,topic,*args,**kwargs):
        self.publishers.append(topic);return Publisher()
    def Subscriber(self,topic,*args,**kwargs):self.subscribers.append(topic)
    def Duration(self,v):return v
    def Timer(self,*args):pass


class PoseStamped(object):
    def __init__(self):
        self.header=Object()
        self.pose=Object(position=Object(x=0.,y=0.,z=0.),orientation=Object(x=0.,y=0.,z=0.,w=0.))


class PlanningNodeTests(unittest.TestCase):
    def test_actual_node_publishes_candidate_path_and_stale_stop(self):
        filename=os.path.join(ROOT,'scripts','planning_shadow_node.py')
        with open(filename,'rb') as stream:tree=ast.parse(stream.read(),filename=filename)
        tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PlanningShadowNode']
        ros=RosDoubles();ros.Time=Object(now=lambda:Object(to_sec=lambda:ros.now))
        scope=dict(rospy=ros,json=json,math=math,threading=threading,PoseStamped=PoseStamped,
                   PoseArray=Object,Path=Path,String=String,
                   PlanningShadowBridge=PlanningShadowBridge,configured_ingress=configured_ingress)
        eval(compile(tree,filename,'exec'),scope)
        node=scope['PlanningShadowNode']();node.tick(None)
        self.assertFalse(json.loads(node.control.messages[-1].data)['valid'])
        node.bridge=bridge()
        for stamp in (1.,1.1):
            ros.now=stamp;node.on_pose(pose_message(stamp));node.on_cones(cone_message(stamp))
            node.on_boundary(String(data=json.dumps(boundary(stamp))));node.tick(None)
        self.assertTrue(json.loads(node.control.messages[-1].data)['valid'])
        self.assertGreater(len(node.path.messages[-1].poses),20)
        ros.now=1.4;node.tick(None)
        out=json.loads(node.control.messages[-1].data)
        self.assertTrue(out['stop_requested']);self.assertFalse(out['vehicle_output'])
        self.assertEqual(node.path.messages[-1].poses,[])
        self.assertTrue(all(p.startswith('~') for p in ros.publishers))


if __name__=='__main__':unittest.main()
