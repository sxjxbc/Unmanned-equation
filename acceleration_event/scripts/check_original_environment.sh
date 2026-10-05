#!/usr/bin/env bash
# Read-only inventory and import checks. No installs, builds, ROS nodes or sockets.
set -u
task_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_package_dir=$(cd -- "$task_script_dir/.." && pwd)
printf '\n=== System ===\n'
cat /etc/os-release
uname -r
printf '\n=== Original ROS/Python baseline ===\n'
command -v python || true
python --version 2>&1 || true
ls -d /opt/ros/* 2>/dev/null || true
if [ ! -f /opt/ros/melodic/setup.bash ]; then
  printf 'STOP: original ROS Melodic is absent. Do not install or substitute another version.\n'
  exit 2
fi
# Only alters this process environment; does not edit user shell configuration.
set +u
source /opt/ros/melodic/setup.bash
set -u
rosversion -d
rosversion roslaunch
/usr/bin/python2 --version 2>&1
printf '\n=== Existing component versions ===\n'
dpkg-query -W -f='${Package}\t${Version}\n' python2.7 python-numpy python-opencv open-vm-tools ros-melodic-roslaunch ros-melodic-rospy ros-melodic-tf 2>/dev/null || true
printf '\n=== Share ===\n'
findmnt -t fuse.vmhgfs-fuse,fuse 2>/dev/null || true
ls -ld /mnt/hgfs/nrt_ws "$task_package_dir"
printf '\n=== Python2 syntax, dependencies and common-module import ===\n'
/usr/bin/python2 - "$task_package_dir" <<'PY'
from __future__ import print_function
import os, sys, hashlib
root = sys.argv[1]
src = os.path.join(root, 'src')
for name in ('data_quality.py', 'jiantu.py', 'zhixian_realtime.py', 'genzong_realtime.py'):
    path = os.path.join(src, name)
    with open(path, 'rb') as stream:
        data = stream.read()
    compile(data, path, 'exec')
    print('SYNTAX OK', name, 'sha256=' + hashlib.sha256(data).hexdigest())
for name in ('numpy', 'cv2', 'rospy', 'tf', 'sensor_msgs.msg', 'nav_msgs.msg', 'geometry_msgs.msg'):
    module = __import__(name, fromlist=['*'])
    print('IMPORT OK', name, getattr(module, '__version__', '(see dpkg version)'))
sys.path.insert(0, src)
import data_quality
print('COMMON MODULE OK', data_quality.monotonic())
PY
task_check_status=$?
if [ "$task_check_status" -ne 0 ]; then
  printf 'STOP: check failed. Preserve component versions and report error.\n'
  exit "$task_check_status"
fi
printf '\n=== Python2 offline tests ===\n'
/usr/bin/python2 -B "$task_package_dir/tests/test_buffer_python2.py"
task_test_status=$?
if [ "$task_test_status" -eq 0 ]; then
  /usr/bin/python2 -B "$task_package_dir/tests/test_centerline_geometry.py"
  task_test_status=$?
fi
if [ "$task_test_status" -eq 0 ]; then
  /usr/bin/python2 -B "$task_package_dir/tests/test_finish_progress.py"
  task_test_status=$?
fi
if [ "$task_test_status" -eq 0 ]; then
  /usr/bin/python2 -B "$task_package_dir/tests/test_lidar_contract.py"
  task_test_status=$?
fi
if [ "$task_test_status" -eq 0 ]; then
  /usr/bin/python2 -B "$task_package_dir/tests/test_localization_mapping.py"
  task_test_status=$?
fi
printf '\nInventory complete. ROS build/replay and vehicle feedback remain unverified.\n'
exit "$task_test_status"
