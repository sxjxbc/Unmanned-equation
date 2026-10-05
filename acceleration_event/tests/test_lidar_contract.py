# -*- coding: utf-8 -*-
from __future__ import print_function
import ast
import os
import unittest
import numpy as np
from test_centerline_geometry import Object, node_methods

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LIDAR = os.path.join(ROOT, 'lidar', 'src', 'lidar_nodes', 'src', 'clustering.py')

def load_functions(names, namespace):
    with open(LIDAR, 'rb') as stream:
        tree = ast.parse(stream.read())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names)
    module = ast.Module(body=body)
    if 'type_ignores' in ast.Module._fields:
        module.type_ignores = []
    eval(compile(ast.fix_missing_locations(module), LIDAR, 'exec'), namespace)
    return namespace

class LidarContractTests(unittest.TestCase):
    def setUp(self):
        self.limits = dict(min_points=5, max_height=.6, max_radius=.3,
                           max_distance=30., min_confidence=.25)
        self.features = dict(N_G=5, H_G=.2, R_G=.1, d=4., C_lidar=.7)
        self.accept = load_functions(['accept_cluster'], dict(np=np))['accept_cluster']

    def test_sparse_height_is_allowed(self):
        self.features['H_G'] = 0.
        self.assertTrue(self.accept(self.features, self.limits))

    def test_reject_outsize_low_quality_and_nonfinite(self):
        for key, value in [('N_G', 4), ('H_G', .7), ('R_G', .31),
                           ('d', 31), ('C_lidar', .2), ('C_lidar', 1.1),
                           ('R_G', float('nan')), ('H_G', -1.)]:
            features = dict(self.features)
            features[key] = value
            self.assertFalse(self.accept(features, self.limits), (key, value))

    def test_publish_preserves_header_confidence_and_empty(self):
        sent = []
        ns = load_functions(['publish_cones'], dict(np=np,
            ConeArray=lambda: Object(header=Object(), cones=[]), Cone=Object,
            publisher_cones=Object(publish=sent.append)))
        ns['publish_cones']({2: [0, 1]}, {2: self.features}, np.array([[1., 2.], [3., 4.]]), 42)
        self.assertEqual(sent[0].header.stamp, 42)
        self.assertEqual(sent[0].header.frame_id, 'velodyne')
        self.assertEqual((sent[0].cones[0].x, sent[0].cones[0].y), (2., 3.))
        self.assertEqual(sent[0].cones[0].confidence, .7)
        ns['publish_cones']({}, {}, np.empty((0, 2)), 43)
        self.assertEqual(sent[1].cones, [])
        self.assertEqual(sent[1].header.stamp, 43)

    def test_empty_cloud_emits_both_contracts(self):
        sent = []
        ns = load_functions(['_clustering_frame'], dict(
            increment_snapshot_counter=lambda: None,
            preprocessing=lambda msg: (42, np.empty((0, 2)), np.array([])),
            rospy=Object(logwarn=lambda *args: None),
            get_PoseArray=lambda *args: args,
            publish_to_perception=lambda msg: sent.append(('legacy', msg)),
            publish_cones=lambda *args: sent.append(('quality', args))))
        ns['_clustering_frame'](Object())
        self.assertEqual([item[0] for item in sent], ['legacy', 'quality'])
        self.assertEqual(sent[1][1][-1], 42)

    def test_quality_callback_rejects_bad_frame_and_invalid_payload(self):
        received = []
        subject = node_methods('jiantu.py', ['quality_cone_callback'], dict(np=np,
            rospy=Object(logwarn_throttle=lambda *args: None),
            PoseArray=lambda: Object(poses=[]),
            Pose=lambda: Object(position=Object())))
        subject.cone_callback = lambda msg, scores: received.append((msg, scores))
        header = Object(frame_id='map')
        subject.quality_cone_callback(Object(header=header, cones=[]))
        self.assertEqual(received, [])
        header.frame_id = '/velodyne'
        subject.quality_cone_callback(Object(header=header, cones=[Object(x=1., y=2., confidence=float('nan'))]))
        self.assertEqual(received, [])
        subject.quality_cone_callback(Object(header=header, cones=[Object(x=1., y=2., confidence=.4)]))
        self.assertEqual(received[0][1], [.4])
        subject.quality_cone_callback(Object(header=header, cones=[]))
        self.assertEqual(received[1][0].poses, [])

if __name__ == '__main__':
    unittest.main()
