#!/usr/bin/env python3
"""
ROS Python3 兼容性包装器
在子进程中用 Python2 运行 ROS 节点，主进程用 Python3
"""
import subprocess
import sys
import os

def main():
    # 设置 Python2 环境
    env = os.environ.copy()
    python2_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
    if python2_path not in env.get('PYTHONPATH', ''):
        env['PYTHONPATH'] = python2_path + ':' + env.get('PYTHONPATH', '')
    
    # 使用 Python2 运行原始脚本
    cmd = ['/usr/bin/python2', 'fake_perception.py'] + sys.argv[1:]
    
    try:
        process = subprocess.Popen(cmd, env=env)
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
