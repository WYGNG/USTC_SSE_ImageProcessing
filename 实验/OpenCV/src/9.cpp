//
// Created by XQ on 2019-04-12.
//
//https://www.kancloud.cn/digest/herbertopencv/100802
#include<iostream>
#include<string>


#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


const double PI =  3.1415926;

class LineFinder
{

private:

    // 存放原图像
    Mat img;
    // 向量中包含检测到的直线的端点
    vector<Vec4i> lines;
    // 累加器的分辨率参数
    double deltaRho;   // 距离
    double deltaTheta; // 角度
    // 被判定为直线所需要的投票数
    int minVote;
    // 直线的最小长度
    double minLength;
    // 沿直线方向的最大缺口
    double maxGap;

public:

    // 默认的累加器分辨率为单个像素，角度为1度
    // 不设置缺口和最小长度的值
    LineFinder() : deltaRho(1), deltaTheta(PI/180), minVote(10), minLength(0.), maxGap(0.) {}
    // 设置累加器的分辨率
    void setAccResolution(double dRho, double dTheta);
    // 设置最小投票数
    void setMinVote(int minv);
    // 设置缺口和最小长度
    void setLineLengthAndGap(double length, double gap);
    // 使用概率霍夫变换
    vector<Vec4i> findLines(Mat& binary);
    // 绘制检测到的直线
    void drawDetectedLines(Mat &image, Scalar color=Scalar(255,255,255));
};
// 设置累加器的分辨率
void LineFinder::setAccResolution(double dRho, double dTheta) {

    deltaRho= dRho;
    deltaTheta= dTheta;
}

// 设置最小投票数
void LineFinder::setMinVote(int minv) {

    minVote= minv;
}

// 设置缺口和最小长度
void LineFinder::setLineLengthAndGap(double length, double gap) {

    minLength= length;
    maxGap= gap;
}

// 使用概率霍夫变换
vector<Vec4i> LineFinder::findLines(Mat& binary)
{

    lines.clear();
    // 调用概率霍夫变换函数
    HoughLinesP(binary,lines,deltaRho,deltaTheta,minVote, minLength, maxGap);
    return lines;
}

// 绘制检测到的直线
void LineFinder::drawDetectedLines(Mat &image, Scalar color)
{
    vector<Vec4i>::const_iterator it2= lines.begin();

    while (it2!=lines.end()) {

        Point pt1((*it2)[0],(*it2)[1]);
        Point pt2((*it2)[2],(*it2)[3]);
        line( image, pt1, pt2, color);

        ++it2;
    }
}

void findLine(string & src){
    Mat image = imread(src,0);
    imshow("Default Image", image);
    // 首先应用Canny算法检测出图像的边缘部分
    Mat contours;
    Canny(image, contours, 125, 350);

    LineFinder finder; // 创建一对象

    // 设置概率Hough参数
    finder.setLineLengthAndGap(100, 20);
    finder.setMinVote(80); //最小投票数

    // 以下步骤检测并绘制直线
    vector<Vec4i>lines = finder.findLines(contours);
    finder.drawDetectedLines(image);

    imshow("Detected Lines", image);

    waitKey(0);
    destroyAllWindows();
}

void findCircle(string & src) {
    // 检测图像中的圆形
    Mat image = imread(src,0);

    // 在调用cv::HoughCircles函数前对图像进行平滑，减少误差
    GaussianBlur(image,image,Size(7,7),1.5);
    vector<Vec3f> circles;
    cv::HoughCircles(image, circles, HOUGH_GRADIENT,
                     2,   // 累加器的分辨率(图像尺寸/2)
                     50,  // 两个圆之间的最小距离
                     200, // Canny中的高阈值
                     100, // 最小投票数
                     20, 80); // 有效半径的最小和最大值

    // 绘制圆圈
    // 一旦检测到圆的向量，遍历该向量并绘制圆形
    // 该方法返回Vec3f类型向量
    // 包含圆圈的圆心坐标和半径三个信息

    vector<Vec3f>::const_iterator itc= circles.begin();
    while (itc!=circles.end())
    {
        circle(image,
               Point((*itc)[0], (*itc)[1]), // 圆心
               (*itc)[2], // 圆的半径
               Scalar(255), // 绘制的颜色
               6); // 圆形的厚度
        ++itc;
    }

    imshow("Detected Circles",image);
    waitKey(0);
    destroyAllWindows();
}
/*
int main(){
        string str = "/Volumes/数据/图片/2k/Lattice.png";
        findLine(str);

        //findCircle(str);
        return 0;
}*/
