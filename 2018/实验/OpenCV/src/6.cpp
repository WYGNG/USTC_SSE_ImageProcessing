//
// Created by XQ on 2019-04-05.
//

//https://www.kancloud.cn/digest/herbertopencv/100797

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

class MorphoFeatures{
public:
    bool fourFunctions(string &src, int flag);
    bool getEdges(const string &src,int flag, int threshold);
    Mat getCorners(const string &src, int flag);
    void drawOnImage(const Mat &binary, Mat &image);
    Mat cross();
    Mat diamond();
    Mat x();
    Mat squre();
};



/*int main(){

    //第一步，先用十字形的结构元素膨胀原图像，这种情况下只会在边缘处“扩张”，角点不发生变化。接着用菱形的结构元素腐蚀原图像，只有拐角处才会被“收缩”，而直线边缘不发生变化。
    //第二步，用X型的结构元素膨胀原图像，角点膨胀的比边要多。这样第二次用方块腐蚀时，角点恢复原状，而边要腐蚀的更多。
    //第三步，将一二步的两幅输出图像相减，结果只保留了各个拐角处的细节。
    //string str = "/Volumes/数据/图片/2k/lostwall.jpg";
    string str = "/Volumes/数据/图片/2k/Lattice.png";
    MorphoFeatures input;
    //
    //input.fourFunctions(str,0);
    //
    input.getEdges(str,0,80);
    //
    Mat corner = input.getCorners(str,0);
    Mat image = imread(str,0);
    input.drawOnImage(corner,image);
    imshow("Corners On Image", image);

    //
    waitKey(0);
    destroyAllWindows();

    return 0;
}*/


bool MorphoFeatures::fourFunctions(string &src, int flag){

    Mat image = imread(src,flag);
    imshow("Default Image", image);
    Mat eroded;
    erode(image, eroded,Mat());
    Mat dilated;
    dilate(image, dilated, Mat());
    Mat closed;
    Mat element1(3, 3, CV_8U, Scalar(1));
    morphologyEx(image,             // 输入图像
                 closed,            // 输出图像
                 MORPH_CLOSE,   // 指定操作
                 element1,          // 结构元素设置
                 Point(-1,-1),  // 操作的位置
                 1);                // 操作的次数
    Mat opened;
    Mat element2(3, 3, CV_8U, Scalar(1));
    morphologyEx(image, opened, MORPH_OPEN, element2, Point(-1,-1), 1);

    imshow("Eroded Image", eroded);
    imshow("Dilated Image", dilated);
    imshow("Closed Image", closed);
    imshow("Opened Image", opened);

    waitKey(0);
    destroyAllWindows();

    return true;
}

bool MorphoFeatures::getEdges(const string &src,int flag, int threshold){
    Mat image = imread(src,flag);
    imshow("Default image", image);
    // 得到梯度图
    Mat result;
    morphologyEx(image, result, MORPH_GRADIENT, Mat());
    // 阈值化以得到二值图像

    // 使用阈值化
    if(threshold > 0){
        cv::threshold(result, result, threshold, 255, THRESH_BINARY);
    }
    imshow("Edge Image",result);

    waitKey(0);
    destroyAllWindows();

    return true;
}

Mat MorphoFeatures::cross() {
    Mat res = Mat(5,5,CV_8U,Scalar(0));
    for (int i=0; i<5; i++) {
        res.at<uchar>(2,i)= 1;
        res.at<uchar>(i,2)= 1;
    }
    return res;

}
Mat MorphoFeatures::diamond() {
    Mat res = Mat(5,5,CV_8U,Scalar(1));
    res.at<uchar>(0,0)= 0;
    res.at<uchar>(0,1)= 0;
    res.at<uchar>(1,0)= 0;
    res.at<uchar>(4,4)= 0;
    res.at<uchar>(3,4)= 0;
    res.at<uchar>(4,3)= 0;
    res.at<uchar>(4,0)= 0;
    res.at<uchar>(4,1)= 0;
    res.at<uchar>(3,0)= 0;
    res.at<uchar>(0,4)= 0;
    res.at<uchar>(0,3)= 0;
    res.at<uchar>(1,4)= 0;
    return res;


}
Mat MorphoFeatures::x() {
    Mat res = Mat(5,5,CV_8U,Scalar(0));
    for (int i=0; i<5; i++)
    {
        res.at<uchar>(i,i)= 1;
        res.at<uchar>(4-i,i)= 1;
    }
    return res;

}
Mat MorphoFeatures::squre() {
    Mat res = Mat(5,5,CV_8U,Scalar(1));
    return res;

}


Mat MorphoFeatures::getCorners(const string& src,int flag){

    Mat image = imread(src,flag);

    Mat result;

    dilate(image, result, this->cross());
    erode(result,result, this->diamond());
    Mat result2;
    dilate(image,result2, this->x());
    erode(result2,result2, this->squre());
    absdiff(result2,result,result);

    cv::threshold(result, result, 80, 255, cv::THRESH_BINARY);
    imshow("Corners",result);
    waitKey(0);
    destroyAllWindows();
    return result;
}

void MorphoFeatures::drawOnImage(const cv::Mat &binary, cv::Mat &image)
{
    cv::Mat_<uchar>::const_iterator it = binary.begin<uchar>();
    cv::Mat_<uchar>::const_iterator itend = binary.end<uchar>();
    for(int i=0; it!=itend; ++it, ++i)
    {
        if(*it) // 若该像素被标定为角点则画白色圈圈
        {
            cv::circle(image, cv::Point(i%image.step, i/image.step), 5, cv::Scalar(255, 255, 255));
        }
    }
}

