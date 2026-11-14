#!/usr/bin/env python3
"""Run selected transport-free regressions and save reviewable evidence."""
import argparse
import ast
import io
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET


def main():
    workspace = Path(__file__).resolve().parents[3]
    package = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(workspace/'exports'/'track_frame_validation_20261015'))
    parser.add_argument('--baseline', default=str(workspace/'backups'/'track_frame_20261015_221841'/'acceleration_event'))
    args = parser.parse_args()
    sys.path.insert(0, str(package/'tests'))
    sys.path.insert(0, str(package/'src'))
    modules = ['test_track_frame_nodes', 'test_legacy_run_guard', 'test_lidar_fusion',
               'test_lidar_fusion_node', 'test_run_diagnostics', 'test_startup_outputs',
               'test_complete_legacy_loop']
    suite = unittest.TestLoader().loadTestsFromNames(modules)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)

    def methods(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        return {node.name: ast.dump(node, include_attributes=False)
                for cls in tree.body if isinstance(cls, ast.ClassDef)
                for node in cls.body if isinstance(node, ast.FunctionDef)}

    baseline = Path(args.baseline)
    old = methods(baseline/'src'/'genzong_realtime.py')
    new = methods(package/'src'/'genzong_realtime.py')
    preserved = {name: old[name] == new[name] for name in (
        'pid_control', 'filter_tracking_errors', 'update_dynamic_lookahead', 'send_steering_frame')}
    old_xml = ET.parse(str(baseline/'launch'/'acceleration_event.launch'))
    new_xml = ET.parse(str(package/'launch'/'acceleration_event.launch'))
    def settings(tree):
        root = tree.getroot()
        values = {'arg/'+a.get('name'):a.get('default') for a in root.findall('arg')}
        for node in root.findall('node'):
            for param in node.findall('param'):
                values[node.get('type')+'/'+param.get('name')] = param.get('value')
        return values
    before, after = settings(old_xml), settings(new_xml)
    preserved['existing_launch_parameters'] = all(after.get(k) == v for k,v in before.items())
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'tests.txt').write_text(output.getvalue(), encoding='utf-8')
    report = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  preserved=preserved, python=sys.version, baseline=str(baseline),
                  limitations=['No ROS transport, sockets or vehicle actuation.',
                               'Geometry tests prescribe poses; added full-loop tests use an uncalibrated kinematic plant.',
                               'Passing regressions do not mean all CLI closed-loop stress scenarios pass; see separate report.',
                               'Selected regressions only, not every historical test in the workspace.',
                               'Python 2.7 and vehicle ROS build were not executed.'])
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() and all(preserved.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
