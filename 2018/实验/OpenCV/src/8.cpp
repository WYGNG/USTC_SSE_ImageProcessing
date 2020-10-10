//图像灰度变换
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

bool Demo1(string & src){
    //缩放
    Mat image;
    image = imread(src, 1); // Read the file

    imshow("Default Image", image);                // Show our image inside it.

    Mat dst = Mat::zeros(256, 256, CV_8UC3); //我要转化为512*512大小的
    resize(image, dst, dst.size());

    imshow("After Modify", dst);

    waitKey(0); // Wait for a keystroke in the window
    destroyAllWindows();
    return true;
}

bool Demo2(string & src,double a, double b){
    Mat image = imread(src, 1);
    imshow("Default Image", image);

    Mat dst;
    resize(image, dst, Size(), a, b);//变为原来的0.5倍

    imshow("After Modify", dst);

    waitKey(0);
    destroyAllWindows();
    return true;
}


bool Demo3(string & src){
    Mat image = imread(src, 1);
    imshow("Default Image", image);

    Mat dst, dst2;
    pyrUp(image, dst, Size(image.cols * 2, image.rows * 2)); //放大一倍
    pyrDown(image, dst2, Size(image.cols * 0.5, image.rows * 0.5)); //缩小为原来的一半

    imshow("After Modify Up", dst);
    imshow("After Modify Down", dst2);

    waitKey(0);
    destroyAllWindows();
    return true;
}

/*
int main()
{
    string str = "/Volumes/数据/图片/2k/6.jpg";
    cout << Demo1(str) << endl;
    cout << Demo2(str, 0.3, 0.3) << endl;
    cout << Demo3(str) << endl;
    return 0;
}*/
