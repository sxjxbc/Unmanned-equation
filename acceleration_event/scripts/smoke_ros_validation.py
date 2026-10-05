#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Isolated ROS startup check; validation launch disables all control outputs."""
from __future__ import print_function
import os
import signal
import socket
import subprocess
import tempfile
import time


def main():
    probe = socket.socket()
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
    probe.close()
    env = os.environ.copy()
    env['ROS_MASTER_URI'] = 'http://127.0.0.1:%d' % port
    env['ROS_HOSTNAME'] = '127.0.0.1'
    env.pop('ROS_IP', None)
    folder = tempfile.mkdtemp(prefix='nrt-ros-smoke-')
    env['ROS_LOG_DIR'] = folder
    children = []
    handles = []
    try:
        for args, name in ((['roscore', '-p', str(port)], 'master'),
                           (['roslaunch', 'acceleration_event', 'validation.launch',
                             'use_sim_time:=false', 'log_dir:=' + folder], 'nodes')):
            handle = open(os.path.join(folder, name + '.log'), 'w')
            handles.append(handle)
            children.append(subprocess.Popen(args, env=env, stdout=handle,
                                             stderr=subprocess.STDOUT, preexec_fn=os.setsid))
            time.sleep(4)
        nodes = subprocess.check_output(['rosnode', 'list'], env=env)
        print(nodes)
        for expected in ('/fsae_slam_node', '/straight_line_planner_realtime', '/path_tracking_controller'):
            assert expected in nodes, 'Missing node: ' + expected
            subprocess.check_call(['rosnode', 'ping', '-c', '1', expected], env=env)
        topics = subprocess.check_output(['rostopic', 'list'], env=env)
        assert '/control/steering' not in topics, 'Unexpected steering publisher'
        assert '/fssim/cmd' not in topics, 'Unexpected simulator command publisher'
        assert all(child.poll() is None for child in children), 'Launch process terminated'
        print('PASS: three live nodes, vehicle command topics absent')
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        for child in children:
            for unused in range(30):
                if child.poll() is not None:
                    break
                time.sleep(.1)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait()
        for handle in handles:
            handle.close()
        print('Logs: ' + folder)


if __name__ == '__main__':
    main()
