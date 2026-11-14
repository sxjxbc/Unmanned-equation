#!/usr/bin/env python3
import subprocess
import os
import sys

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
jiantu_path = os.path.join(script_dir, 'jiantu.py')

# 检查jiantu.py是否存在
if not os.path.exists(jiantu_path):
    print(f"错误: 找不到 jiantu.py 在 {jiantu_path}")
    sys.exit(1)

# 设置 Python2 环境
env = os.environ.copy()
env['PYTHONPATH'] = '/opt/ros/melodic/lib/python2.7/dist-packages:' + env.get('PYTHONPATH', '')
env['DISPLAY'] = ':0'
env['MPLBACKEND'] = 'Agg'

# 使用 Python2 运行 jiantu.py
cmd = ['/usr/bin/python2', jiantu_path] + sys.argv[1:]

print(f"使用Python2运行: {jiantu_path}")
print(f"命令: {' '.join(cmd)}")

try:
    process = subprocess.Popen(cmd, env=env)
    process.wait()
except KeyboardInterrupt:
    print("收到中断信号，停止程序...")
    process.terminate()
    process.wait()
    sys.exit(0)
except Exception as e:
    print(f"运行错误: {e}")
    sys.exit(1)
