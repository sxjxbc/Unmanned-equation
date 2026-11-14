#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/melodic/setup.bash
source /home/nvidia/nrt_ws/devel/setup.bash
source /home/nvidia/nrt_release_20261019/devel/setup.bash --extend
mkdir -p /home/nvidia/straight_acceleration_logs
exec roslaunch acceleration_event acceleration_event.launch enable_vehicle_output:=true enable_visualization:=false log_dir:=/home/nvidia/straight_acceleration_logs
