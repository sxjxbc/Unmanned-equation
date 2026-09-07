#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cv_bridge/cv_bridge.h>

// ===================== 数据结构：锥桶检测结果 =====================
struct ConeDetectionResult
{
    std::string label;       // "red" "yellow" "blue"
    cv::Point2f center;      //像素中心点
    cv::Rect bbox;          // bounding box
    float confidence;
};

// ===================== 1.自适应伽马校正 =====================
cv::Mat adaptiveGammaCorrection(const cv::Mat& src)
{
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    double meanBright = cv::mean(gray)[0];

    double gamma = 1.2 - meanBright / 128.0;
    gamma = cv::max(0.4, cv::min(1.8, gamma));

    // 查表加速
    cv::Mat lookUpTable(1,256,CV_8UC1);
    uchar* p = lookUpTable.data;
    for(int i = 0; i < 256; i++)
    {
        p[i] = static_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::Mat dst;
    cv::LUT(src, lookUpTable, dst);
    return dst;
}

// ===================== 2.从mask提取锥桶轮廓 =====================
std::vector<ConeDetectionResult> getConesFromMask(const cv::Mat& mask,
                                                   const std::string& label,
                                                   double minArea = 80.0)
{
    std::vector<ConeDetectionResult> res;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // OpenCV3.2接口
    cv::findContours(mask.clone(), contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for(auto &cnt : contours)
    {
        double area = cv::contourArea(cnt);
        if(area < minArea)
            continue;

        cv::Rect rect = cv::boundingRect(cnt);
        ConeDetectionResult cone;
        cone.label = label;
        cone.bbox = rect;
        cone.center.x = rect.x + rect.width * 0.5f;
        cone.center.y = rect.y + rect.height * 0.5f;
        cone.confidence = 1.0f;
        res.push_back(cone);
    }
    return res;
}

// ===================== 3.整套传统检测入口函数 =====================
std::vector<ConeDetectionResult> traditionalConeDetect(const cv::Mat& src)
{
    std::vector<ConeDetectionResult> detections;

    //步骤1：自适应伽马
    cv::Mat gammaImg = adaptiveGammaCorrection(src);

    //步骤2：双边滤波去扬尘噪声
    cv::Mat blurImg;
    cv::bilateralFilter(gammaImg, blurImg, 9, 75, 75);

    //步骤3：转到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(blurImg, hsv, cv::COLOR_BGR2HSV);

    //黄色阈值
    cv::Mat maskYellow;
    cv::inRange(hsv, cv::Scalar(18,60,60), cv::Scalar(35,255,255), maskYellow);

    //红色是两段
    cv::Mat maskR1, maskR2, maskRed;
    cv::inRange(hsv, cv::Scalar(0,120,80), cv::Scalar(6,255,255), maskR1);
    cv::inRange(hsv, cv::Scalar(174,120,80), cv::Scalar(180,255,255), maskR2);
    cv::bitwise_or(maskR1, maskR2, maskRed);

    //蓝色阈值
    cv::Mat maskBlue;
    cv::inRange(hsv, cv::Scalar(100,80,60), cv::Scalar(130,255,255), maskBlue);

    //分别提取三种颜色锥桶
    auto yellow = getConesFromMask(maskYellow, "yellow",80.0);
    auto red    = getConesFromMask(maskRed,    "red",   80.0);
    auto blue   = getConesFromMask(maskBlue,   "blue",  80.0);

    detections.insert(detections.end(), yellow.begin(), yellow.end());
    detections.insert(detections.end(), red.begin(), red.end());
    detections.insert(detections.end(), blue.begin(), blue.end());

    return detections;
}

// =====================主函数=====================
int main(int argc, char **argv)
{
    ros::init(argc, argv, "cone_traditional_node");
    ros::NodeHandle nh;

    //打开摄像头，/dev/video0 左相机，/dev/video1右相机
    cv::VideoCapture cap_left(0);
    cv::VideoCapture cap_right(1);

    if(!cap_left.isOpened() || !cap_right.isOpened())
    {
        ROS_ERROR("相机打开失败！");
        return -1;
    }

    ros::Rate rate(10); //10Hz输出
    cv::Mat img_left, img_right;

    while(ros::ok())
    {
        cap_left >> img_left;
        cap_right >> img_right;

        if(img_left.empty() || img_right.empty())
        {
            rate.sleep();
            continue;
        }

        // =========核心调用：传统图像处理检测锥桶=========
        std::vector<ConeDetectionResult> left_result  = traditionalConeDetect(img_left);
        std::vector<ConeDetectionResult> right_result = traditionalConeDetect(img_right);

        // =========这里之后做【双目匹配 + 测距】=========
        // 后面我们在这里移植 calculate_coordinates 测距代码
        // 把得到的锥桶三维位置组装为 msgs::ConeArray 发布出去

        ROS_INFO("左图检测到 %lu 个锥桶", left_result.size());

        ros::spinOnce();
        rate.sleep();
    }

    cap_left.release();
    cap_right.release();
    return 0;
}
