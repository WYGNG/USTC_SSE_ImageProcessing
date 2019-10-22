//
// Created by XQ on 2019-04-05.
//

//https://www.kancloud.cn/digest/herbertopencv/100801

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

class MyCanny{
private:
    Mat image;
    Mat sobel;
    int threshold;
    Mat sobelMagnitude;
    Mat sobelOrientation;

public:
    void setthreshold(int a){
        threshold= a;
    }

    // 获取阈值
    int getthreshold() const{
        return threshold;
    }

    // 计算Sobel结果
    void computeSobel(const Mat &image){
        Mat sobelX;
        Mat sobelY;
        Sobel(image,sobelX,CV_32F,1,0,threshold);
        Sobel(image,sobelY,CV_32F,0,1,threshold);
        cartToPolar(sobelX,sobelY,sobelMagnitude,sobelOrientation);
    }

    // 获取幅度
    Mat getMagnitude(){
        return sobelMagnitude;
    }

    // 获取Sobel方向
    Mat getOrientation(){
        return sobelOrientation;
    }

    // 输入门限获取二值图像
    Mat getBinaryMap(double Threhhold){
        Mat bgImage;
        cv::threshold(sobelMagnitude,bgImage,Threhhold,255,THRESH_BINARY_INV);
        return bgImage;
    }

    // 转化为CV_8U图像
    Mat getSobelImage(){
        Mat bgImage;
        double minval,maxval;
        minMaxLoc(sobelMagnitude,&minval,&maxval);
        sobelMagnitude.convertTo(bgImage,CV_8U,255/maxval);
        return bgImage;
    }

    // 获取梯度
    Mat getSobelOrientationImage(){
        Mat bgImage;
        sobelOrientation.convertTo(bgImage,CV_8U,90/PI);
        return bgImage;
    }
};

/*int main(){

    string str = "/Volumes/数据/图片/2k/Lattice.png";
    Mat image = imread(str,0);
    imshow("Default Image",image);
    auto * mycanny = new MyCanny();
    mycanny->computeSobel(image);
    // 获取sobel的大小和方向
    imshow("Orientation of sobel",mycanny->getSobelOrientationImage());
    imshow("Magnitude of sobel",mycanny->getSobelImage());

    // 使用两种阈值的检测结果
    imshow("The Lower Threshold",mycanny->getBinaryMap(125));
    imshow("The Higher Threshold",mycanny->getBinaryMap(225));

    // 使用canny算法
    Mat contours;
    Canny(image,contours,125,225);
    Mat contoursInv;
    cv::threshold(contours,contoursInv,128,255,cv::THRESH_BINARY_INV);
    imshow("Edge Image",contoursInv);
    waitKey(0);
    destroyAllWindows();

    return 0;
}*/
