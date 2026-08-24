#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <cmath>

ros::Publisher path_pub;

void clustered_callback(const geometry_msgs::PoseArray::ConstPtr& msg);
void print_cone_time_and_number_information(double time,int cone_number);
void print_cone_distance_information( double cone_distance );

double road_center;
double last_road_center = 0.0;
double last_stamp_sec = 0.0;
const double alpha = 0.25;         
const double road_width = 3.0;      
const double path_step = 1.0;       
const double total_track_len = 100000; 
int  main(int argc, char *argv[]){
    ros::init(argc,argv,"acceleration_event_node");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("clustered_points",2,clustered_callback);
     path_pub= nh.advertise<nav_msgs::Path>("/planned_path", 10);
    printf("acceleration_event_node is ready\n");
    while(ros::ok()){
        ros::spinOnce();
    }
    return 0;
}

void clustered_callback(const geometry_msgs::PoseArray::ConstPtr& msg){
    double stamp = msg -> header.stamp.toSec();
    int cone_number = msg -> poses.size();
    print_cone_time_and_number_information(stamp , cone_number);
    double left_min_distance=0.0;
    double right_min_distance=0.0;
    for(int i = 0 ; i < cone_number;i++){
        double x = msg -> poses[i].position.x;
        double y = msg -> poses[i].position.y;
        double cone_distance_square=x*x+y*y;
        if(y<0){
            if(y<right_min_distance){
                right_min_distance=y;
            }
        else if(y>0){
            if(y>left_min_distance){
                left_min_distance=y;
            }
        }
        else{
            continue;
        }
        }
    }
    if (right_min_distance==0){
        road_center = (-road_width/2) +  left_min_distance;
    }
    else if(left_min_distance==0){
        road_center = (road_width/2) + right_min_distance; 
    }
    else {
        road_center = (left_min_distance+right_min_distance)/2;
    }
double dt = stamp - last_stamp_sec;
double smooth_center;
if(last_stamp_sec == 0.0 || dt <= 0 || dt > 1.0)
{
    smooth_center = road_center;
}
else
{
    smooth_center = alpha * road_center + (1 - alpha) * last_road_center;
}
last_road_center = smooth_center;
last_stamp_sec = stamp;
printf("raw_center:%.3f smooth_center:%.3f\n", road_center, smooth_center);
nav_msgs::Path path_msg;
path_msg.header = msg->header;
path_msg.header.frame_id = "velodyne";
for(double x_pos = 0; x_pos <= total_track_len; x_pos += path_step)
{
    geometry_msgs::PoseStamped pose_stamp;
    pose_stamp.header = path_msg.header;
    pose_stamp.pose.position.x = x_pos;
    pose_stamp.pose.position.y = smooth_center;
    pose_stamp.pose.position.z = 0.0;
    pose_stamp.pose.orientation.w = 1.0;
    path_msg.poses.push_back(pose_stamp);
}
path_pub.publish(path_msg);

 }


void print_cone_time_and_number_information(double time,int cone_number){
        printf("stamp: %lf",time);
        printf(" cone_number = %d\n",cone_number);
}

void print_cone_distance_information( double cone_distance ){
        printf("cone_distance = %lf",cone_distance);
}

