#!/bin/bash
set -e
source /opt/ros/melodic/setup.bash
source /home/nvidia/nrt_ws/devel/setup.bash
source /home/nvidia/ebs_test_20261027/devel/setup.bash
source /home/nvidia/trackdrive_continuous_20261103_d63c0976/devel/setup.bash --extend
exec roslaunch obstacle_avoidance_event low_speed_trial.launch output_mode:=vehicle operator_ack:=confirmed_low_speed_test external_speed_limited:=true can_dest_ip:=192.168.0.7 can_dest_port:=20005 can_self_port:=11312
