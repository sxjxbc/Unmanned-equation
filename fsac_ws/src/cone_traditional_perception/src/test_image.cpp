#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

/**
 * @brief 锥桶视觉检测输出结构体
 * contour：原始分割轮廓，用于置信计算；bbox仅可视化
 * center：轮廓矩计算得到的质心像素坐标，后续用于反投影世界坐标
 */
struct ConeDetectionResult
{
    std::string label;                   // "red" "yellow" "blue"
    cv::Point2f center;                  // 轮廓质心 (u,v)
    cv::Rect bbox;                       // 仅可视化，不参与算法计算
    std::vector<cv::Point> contour;      // 原始轮廓多边形
    float confidence;                    // 视觉置信度 C_cam [0,1]
};

/**
 * @brief 不同颜色锥桶HSV参考参数
 */
struct HsvRefParam
{
    int Hc;
    int dHc;
    int Sc;
    int dSc;
    int Vc;
    int dVc;
    double wH;
    double wS;
    double wV;
};

/**
 * @brief 根据类别返回对应HSV参考参数
 */
HsvRefParam getHsvParamByLabel(const std::string& label)
{
    HsvRefParam p;
    // 论文公式(6)权重 w_H=0.50、w_S=0.30、w_V=0.20
    p.wH = 0.50;
    p.wS = 0.30;
    p.wV = 0.20;

    if (label == "red")
    {
        p.Hc = 5;
        p.dHc = 12;
        p.Sc = 180;
        p.dSc = 80;
        p.Vc = 200;
        p.dVc = 90;
    }
    else if (label == "yellow")
    {
        p.Hc = 26;
        p.dHc = 10;
        p.Sc = 170;
        p.dSc = 75;
        p.Vc = 210;
        p.dVc = 90;
    }
    else if (label == "blue")
    {
        p.Hc = 110;
        p.dHc = 15;
        p.Sc = 160;
        p.dSc = 80;
        p.Vc = 180;
        p.dVc = 85;
    }
    return p;
}

/**
 * @brief 论文公式：计算轮廓区域HSV颜色置信度 C_HSV(R)
 * @param hsvImg 输入HSV图像 OpenCV H:0~179 S:0~255 V:0~255
 * @param contour 目标原始轮廓
 * @param param 该颜色锥桶HSV参考参数
 * @return 置信度 [0, 1]
 */
double calcHsvConfidence(const cv::Mat& hsvImg,
    const std::vector<cv::Point>& contour,
    const HsvRefParam& param)
{
    cv::Mat maskRoi = cv::Mat::zeros(hsvImg.size(), CV_8UC1);
    cv::drawContours(maskRoi, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), -1);

    std::vector<cv::Point> roiPoints;
    cv::findNonZero(maskRoi, roiPoints);
    int pixelNum = roiPoints.size();
    if (pixelNum <= 0)
        return 0.0;

    double sumScore = 0.0;
    int validpixelNum = 0;

    int Hc = param.Hc;
    int dHc = param.dHc;
    int Sc = param.Sc;
    int dSc = param.dSc;
    int Vc = param.Vc;
    int dVc = param.dVc;
    double wH = param.wH;
    double wS = param.wS;
    double wV = param.wV;

    for (const auto& pt : roiPoints)
    {
        cv::Vec3b hsv = hsvImg.at<cv::Vec3b>(pt);
        int H = hsv[0];
        int S = hsv[1];
        int V = hsv[2];

        // 强光过曝白色像素，跳过不参与计算
        if (V > 240 && S < 40)
        {
            continue;
        }

        // 色相环形距离，兼容红色跨0边界
        int dhRaw = std::abs(H - Hc);
        int dh = std::min(dhRaw, 180 - dhRaw);
        double scoreH = wH * std::max(0.0, 1.0 - static_cast<double>(dh) / static_cast<double>(dHc));

        double ds = std::abs(S - Sc);
        double scoreS = wS * std::max(0.0, 1.0 - ds / static_cast<double>(dSc));

        double dv = std::abs(V - Vc);
        double scoreV = wV * std::max(0.0, 1.0 - dv / static_cast<double>(dVc));

        sumScore += (scoreH + scoreS + scoreV);
        validpixelNum++;
    }

    // 轮廓全部是过曝白色像素，工程宽容处理，返回0.6，防止除零崩溃
    if (validpixelNum <= 0)
    {
        return 0.60;
    }

    double cHsv = sumScore / static_cast<double>(validpixelNum);
    return cHsv;
}

/**
 * @brief 自适应Gamma校正
 */
cv::Mat adaptiveGammaCorrection(const cv::Mat& src)
{
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    double meanBright = cv::mean(gray)[0];
    double meanNorm = meanBright / 255.0;
    const double eps = 1e-6;
    double gamma = std::log10(0.5) / std::log10(meanNorm + eps);
    gamma = cv::max(0.4, cv::min(2.5, gamma));

    cv::Mat lookUpTable(1, 256, CV_8UC1);
    uchar* p = lookUpTable.data;
    for (int i = 0; i < 256; i++)
    {
        p[i] = static_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }
    cv::Mat dst;
    cv::LUT(src, lookUpTable, dst);
    return dst;
}

/**
 * @brief 从mask提取轮廓，基于原始轮廓计算全套视觉置信度
 * @param mask 二值分割mask
 * @param hsv 完整hsv图像，用于颜色置信计算
 * @param label 锥桶类别 red/yellow/blue
 * @return 检测候选列表
 */
std::vector<ConeDetectionResult> getConesFromMask(const cv::Mat& mask,
    const cv::Mat& hsv,
    const std::string& label)
{
    std::vector<ConeDetectionResult> res;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    HsvRefParam hsvParam = getHsvParamByLabel(label);

    // 论文公式(7)加权系数 λ1+λ2+λ3 =1
    const double lambda1 = 0.70;
    const double lambda2 = 0.20;
    const double lambda3 = 0.10;
    // 置信判决阈值Tc
    const double Tc = 0.55;

    for (auto& cnt : contours)
    {
        double area = cv::contourArea(cnt);
        // 面积阈值，过滤过小噪声
        if (area < 80)
            continue;

        cv::Rect rectVis = cv::boundingRect(cnt);
        cv::RotatedRect rotRect = cv::minAreaRect(cnt);
        float len1 = rotRect.size.width;
        float len2 = rotRect.size.height;
        float hReal = std::max(len1, len2);
        float wReal = std::min(len1, len2);
        float ratio = hReal / (wReal + 1e-6f);

        // ========== 1.颜色置信 C_HSV ==========
        double cHsv = calcHsvConfidence(hsv, cnt, hsvParam);

        // ========== 2.面积合理性置信 C_A(R) [0~1] ==========
        double cA = 1.0;
        if (area < 120)
        {
            cA = 0.25;
        }
        else if (area > 12000)
        {
            cA = 0.40;
        }

        // ==========3.形状质量置信 C_Q(R)：融合高宽比惩罚+凸包密实度 ==========
        float shapeScore = 1.0f;
        if (ratio < 1.0f || ratio > 3.0f)
        {
            shapeScore *= 0.45f;
        }
        else if (ratio < 1.2f || ratio > 2.6f)
        {
            shapeScore *= 0.75f;
        }
        std::vector<cv::Point> hull;
        cv::convexHull(cnt, hull);
        double areaHull = cv::contourArea(hull);
        double solidity = area / (areaHull + 1e-6);
        double cQ = static_cast<double>(shapeScore) * solidity;

        // ==========论文公式(7)加权求和得到最终置信度==========
        double finalConf = lambda1 * cHsv + lambda2 * cA + lambda3 * cQ;
        // 钳位 0~1
        finalConf = std::max(0.0, std::min(1.0, finalConf));

        // 置信阈值判决，低于Tc直接舍弃该候选
        if (finalConf < Tc)
        {
            continue;
        }

        // 轮廓矩求解质心
        cv::Moments mu = cv::moments(cnt);
        double cx = mu.m10 / (mu.m00 + 1e-6);
        double cy = mu.m01 / (mu.m00 + 1e-6);

        ConeDetectionResult cone;
        cone.label = label;
        cone.bbox = rectVis;
        cone.contour = cnt;
        cone.center = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
        cone.confidence = static_cast<float>(finalConf);
        res.push_back(cone);
    }
    return res;
}

/**
 * @brief 完整视觉锥桶检测入口
 * @param src 输入BGR原始图像
 * @return 全部锥桶候选
 */
std::vector<ConeDetectionResult> detectConeImage(const cv::Mat& src)
{
    std::vector<ConeDetectionResult> total;
    // 1 自适应Gamma光照补偿
    cv::Mat gammaImg = adaptiveGammaCorrection(src);
    // 2 双边滤波，保边缘抑噪声
    cv::Mat blurImg;
    cv::bilateralFilter(gammaImg, blurImg, 9, 75, 75);
    // 3 转HSV
    cv::Mat hsv;
    cv::cvtColor(blurImg, hsv, cv::COLOR_BGR2HSV);

    // --------黄色锥桶掩码--------
    cv::Mat maskYellow;
    cv::inRange(hsv, cv::Scalar(18, 80, 60), cv::Scalar(35, 255, 255), maskYellow);
    // --------蓝色锥桶掩码--------
    cv::Mat maskBlue;
    cv::inRange(hsv, cv::Scalar(95, 80, 40), cv::Scalar(130, 255, 255), maskBlue);
    // --------红色锥桶掩码（两段阈值）--------
    cv::Mat maskR1, maskR2, maskRed;
    cv::inRange(hsv, cv::Scalar(0, 90, 80), cv::Scalar(8, 255, 255), maskR1);
    cv::inRange(hsv, cv::Scalar(160, 90, 80), cv::Scalar(179, 255, 255), maskR2);
    maskRed = maskR1 | maskR2;

    // 提取各类候选
    auto yellowCones = getConesFromMask(maskYellow, hsv, "yellow");
    auto redCones = getConesFromMask(maskRed, hsv, "red");
    auto blueCones = getConesFromMask(maskBlue, hsv, "blue");

    total.insert(total.end(), yellowCones.begin(), yellowCones.end());
    total.insert(total.end(), redCones.begin(), redCones.end());
    total.insert(total.end(), blueCones.begin(), blueCones.end());
    return total;
}

/**
 * @brief 离线图片测试主函数
 */
int main(int argc, char** argv)
{
    ros::init(argc, argv, "cone_vision_offline");
    ros::NodeHandle nh;
    std::string imgPath = "/home/lty/cone.jpg";
    cv::Mat image = cv::imread(imgPath);
    if (image.empty())
    {
        ROS_ERROR("图片读取失败，路径：%s", imgPath.c_str());
        return -1;
    }
    std::vector<ConeDetectionResult> results = detectConeImage(image);

    cv::Mat drawImg = image.clone();
    for (auto& cone : results)
    {
        cv::Scalar drawColor;
        if (cone.label == "red")
            drawColor = cv::Scalar(0, 0, 255);
        else if (cone.label == "yellow")
            drawColor = cv::Scalar(0, 255, 255);
        else if (cone.label == "blue")
            drawColor = cv::Scalar(255, 0, 0);

        cv::drawContours(drawImg, std::vector<std::vector<cv::Point>>{cone.contour}, 0, drawColor, 2);
        cv::circle(drawImg, cone.center, 4, cv::Scalar(0, 255, 0), -1);
        std::string text = cone.label + " " + std::to_string(cone.confidence).substr(0, 4);
        cv::putText(drawImg, text, cv::Point(cone.bbox.x, cone.bbox.y - 6),
            cv::FONT_HERSHEY_SIMPLEX, 0.42, drawColor, 1);

        ROS_INFO("Detect: label=%s conf=%.3f centroid=(%.1f,%.1f)",
            cone.label.c_str(), cone.confidence, cone.center.x, cone.center.y);
    }

    cv::imshow("cone_detect_result", drawImg);
    cv::imwrite("./cone_result_out.jpg", drawImg);
    ROS_INFO("输出图片保存到 ./cone_result_out.jpg, total cones: %lu", results.size());
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
