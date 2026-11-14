#!/bin/bash
set -e
source /opt/ros/melodic/setup.bash
source /home/nvidia/nrt_ws/devel/setup.bash
source /home/nvidia/ebs_test_20261027/devel/setup.bash
source /home/nvidia/trackdrive_continuous_20261103_d63c0976/devel/setup.bash --extend
while [ "$(cat /sys/class/net/eth0/carrier 2>/dev/null)" != "1" ]; do echo "Waiting for lidar eth0 link"; sleep 1; done
exec roslaunch obstacle_avoidance_event trial_lidar.launch
