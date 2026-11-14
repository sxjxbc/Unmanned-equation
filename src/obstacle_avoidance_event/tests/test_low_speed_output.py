import os,sys,unittest,ast,math,socket,json,threading
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(ROOT,'src'))
from obstacle_avoidance_event.low_speed_output import LowSpeedOutputGuard


class Client(object):
    def __init__(self):self.requests=[];self.closed=False;self.fail=False
    def send_steering_frame(self,angle,stop=False):
        self.requests.append((angle,stop));return not self.fail
    def close(self):self.closed=True


def packet(stamp=1.,x=0.):
    return dict(valid=True,drive_eligible=True,source_stamp=stamp,steering=.1,
                speed_limit=.5,estimated_speed=.2,rear_position=[x,0.])


class OutputGuardTests(unittest.TestCase):
    def test_ros_node_default_does_not_construct_transport(self):
        filename=os.path.join(ROOT,'scripts','low_speed_output_node.py')
        with open(filename,'rb') as f:tree=ast.parse(f.read(),filename=filename)
        tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OutputNode']
        class Object(object):
            def __init__(self,**kw):self.__dict__.update(kw)
        class Message(object):
            def __init__(self,data):self.data=data
        class Ros(object):
            def get_param(self,key,default):return default
            def Publisher(self,*a,**k):return Object(publish=lambda m:None)
            def Subscriber(self,*a,**k):pass
            def Timer(self,*a,**k):pass
            def Duration(self,value):return value
            def on_shutdown(self,callback):self.shutdown=callback
            Time=Object(now=lambda:Object(to_sec=lambda:1.))
        called=[]
        scope=dict(rospy=Ros(),threading=threading,json=json,String=Message,
                   LowSpeedOutputGuard=LowSpeedOutputGuard,original_client=lambda mode:called.append(mode))
        eval(compile(tree,filename,'exec'),scope)
        node=scope['OutputNode']();node.tick(None);node.shutdown()
        self.assertEqual(called,[])

    def test_disabled_never_sends_including_shutdown(self):
        c=Client();g=LowSpeedOutputGuard(c,enabled=False)
        g.receive(packet(),1.);g.tick(1.);g.shutdown()
        self.assertEqual(c.requests,[]);self.assertFalse(c.closed)

    def test_current_shadow_not_eligible_to_send(self):
        c=Client();g=LowSpeedOutputGuard(c,enabled=True)
        p=packet();p['drive_eligible']=False;g.receive(p,1.)
        self.assertEqual(g.tick(1.)['action'],'wait');self.assertEqual(c.requests,[])

    def test_steering_step_and_timeout_latch(self):
        c=Client();g=LowSpeedOutputGuard(c,enabled=True)
        g.receive(packet(),1.);self.assertAlmostEqual(g.tick(1.)['steering'],.012)
        g.receive(packet(),1.4) # Republishing cannot refresh source age.
        self.assertEqual(g.tick(1.6)['action'],'stop')
        g.receive(packet(1.7),1.7);self.assertEqual(g.tick(1.7)['action'],'stop')

    def test_missing_source_and_explicit_stop(self):
        for p in ({'valid':True,'drive_eligible':True},dict(stop_requested=True)):
            c=Client();g=LowSpeedOutputGuard(c,enabled=True);g.receive(p,1.)
            self.assertEqual(g.tick(1.)['action'],'stop')
            self.assertTrue(c.requests[-1][1])

    def test_distance_and_repeated_pose(self):
        c=Client();g=LowSpeedOutputGuard(c,enabled=True,distance_limit=.2)
        g.receive(packet(),1.);g.tick(1.);g.tick(1.05)
        self.assertEqual(g.progress,0.)
        g.receive(packet(1.2,.2),1.2)
        self.assertEqual(g.tick(1.2)['stop_reason'],'test_distance_complete')

    def test_duration_progress_jump_and_conflict(self):
        c=Client();g=LowSpeedOutputGuard(c,enabled=True,duration_limit=.5)
        g.receive(packet(),1.);g.tick(1.)
        self.assertEqual(g.tick(1.6)['stop_reason'],'test_duration_complete')
        c=Client();g=LowSpeedOutputGuard(c,enabled=True);g.receive(packet(),1.);g.tick(1.)
        g.receive(packet(1.1,3.),1.1);self.assertEqual(g.tick(1.1)['stop_reason'],'test_position_jump')
        c=Client();g=LowSpeedOutputGuard(c,enabled=True)
        self.assertEqual(g.tick(1.,channel_conflict=True)['action'],'stop')

    def test_continuous_trial_exceeds_old_limits_and_fault_still_latches(self):
        c=Client();g=LowSpeedOutputGuard(c,enabled=True,distance_limit=0.,duration_limit=0.)
        for i in range(181):
            now=1.+i*.1
            g.receive(packet(now,i*.025),now)
            self.assertEqual(g.tick(now)['action'],'steer')
        self.assertGreater(g.progress,3.)
        self.assertEqual(g.tick(19.6)['action'],'stop')
        self.assertEqual(g.fault,'output_candidate_stale')
        g.receive(packet(19.7,4.5),19.7)
        self.assertEqual(g.tick(19.7)['action'],'stop')
        self.assertTrue(c.requests[-1][1])

    def test_overspeed_send_failure_and_shutdown(self):
        for key,value in (('estimated_speed',2.),('steering',.8),('estimated_speed',float('nan'))):
            c=Client();g=LowSpeedOutputGuard(c,enabled=True);p=packet();p[key]=value;g.receive(p,1.)
            self.assertEqual(g.tick(1.)['action'],'stop')
        c=Client();c.fail=True;g=LowSpeedOutputGuard(c,enabled=True);g.receive(packet(),1.)
        self.assertEqual(g.tick(1.)['stop_reason'],'steering_send_failed')
        g.shutdown();self.assertTrue(c.closed);self.assertTrue(c.requests[-1][1])

    def test_original_codec_udp_loopback_and_exclusive_bind(self):
        # Reuse the actual codec and exclusive wrapper, never a vehicle address.
        filename=os.path.join(ROOT,'..','acceleration_event','src','genzong_realtime.py')
        with open(filename,'rb') as f:tree=ast.parse(f.read(),filename=filename)
        tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CANUDPClient']
        class Logger(object):
            def __getattr__(self,name):return lambda *args:None
        scope=dict(math=math,rospy=Logger(),CAN_ID_STEERING=0x210)
        eval(compile(tree,filename,'exec'),scope)
        other=os.path.join(ROOT,'..','ebs_test','scripts','ebs_output.py')
        with open(other,'rb') as f:code=compile(f.read(),other,'exec')
        wrapper={};eval(code,wrapper)
        receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);receiver.bind(('127.0.0.1',0));receiver.settimeout(2)
        factory=wrapper['exclusive_client_class'](scope['CANUDPClient'])
        client=factory('127.0.0.1',receiver.getsockname()[1],0)
        try:
            client.init();source=client.sock.getsockname()[1]
            second=factory('127.0.0.1',receiver.getsockname()[1],source)
            with self.assertRaises(Exception):second.init()
            g=LowSpeedOutputGuard(client,enabled=True);g.receive(packet(),1.)
            self.assertEqual(g.tick(1.)['action'],'steer')
            normal=bytearray(receiver.recv(64));self.assertEqual(list(normal[:5]),[8,0,0,2,16])
            self.assertEqual(normal[5]+256*normal[6],int(round(500+.012*1000/math.pi)))
            g.tick(2.);stop=bytearray(receiver.recv(64));self.assertEqual(stop[5]+256*stop[6],2000)
        finally:client.close();receiver.close()


if __name__=='__main__':unittest.main()
