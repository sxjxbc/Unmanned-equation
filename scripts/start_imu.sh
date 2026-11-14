#!/bin/bash
set -e
source /opt/ros/melodic/setup.bash
source /home/nvidia/nrt_ws/devel/setup.bash
source /home/nvidia/ebs_test_20261027/devel/setup.bash
source /home/nvidia/trackdrive_continuous_20261103_d63c0976/devel/setup.bash --extend
exec rosrun obstacle_avoidance_event wait_for_imu.py
