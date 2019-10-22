//
// Created by XQ on 2019-03-28.
//

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


bool loadImage(string &, int);
bool binaryImage(string &, int flag = 1, int r = 80);
bool logTransformImage(string &, int flag = 1, double r = 5);
bool gammaImage(string &, int flag = 1, double r = 1.2);
bool complementColorImage(string &, int flag = 1);
/*int main(){
    string str = "/Volumes/数据/图片/2k/lostwall.jpg";
    cout << loadImage(str,1) << endl;
    cout << binaryImage(str,0) << endl;
    cout << logTransformImage(str,0) <<endl;
    cout << gammaImage(str,0) << endl;
    cout << complementColorImage(str) << endl;
}*/

/*利用 OpenCV 读取图像。
 *
 * 具体内容
 *  用打开 OpenCV 打开图像，并在窗口中显示
 *
 * */
bool loadImage(string &src, int flag = 1){
    Mat img = imread(src, flag);
    imshow("simpleOpenImage",img);
    waitKey(0);
    destroyAllWindows();
    return true;
}

/*灰度图像二值化处理
 *
 * 具体内容
 *  设置并调整阈值对图像进行二值化处理。
 *
 * */
bool binaryImage(string &src, int flag, int r){
    Mat image, res;
    image = imread(src, flag);
    imshow("inputImage", image);

    //用于二值化的图像
    res = image.clone();
    int rows = res.rows;
    int cols = res.cols;

    //二值化
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            auto gray = res.at<uchar>(i, j);
            if (gray > r)   gray = 255;
            else    gray = 0;
            res.at<uchar>(i, j) = saturate_cast<uchar>(gray);
        }
    }
    imshow("binaryImage", res);
    waitKey(0);
    destroyAllWindows();
    return true;

}

/*灰度图像的对数变换
 *
 * 具体内容：
 *  设置并调整 r 值对图像进行对数变换。
 *
 * */
bool logTransformImage(string &src, int flag, double r){
    Mat image, res;
    image = imread(src, flag);
    imshow("inputImage", image);

    //用于对数变换
    res = image.clone();
    int rows = res.rows;
    int cols = res.cols;

    //对数变换
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            auto gray = (double)res.at<uchar>(i, j);
            gray = r * log((double)(1 + gray));
            res.at<uchar>(i, j) = saturate_cast<uchar>(gray);
        }
    }
    normalize(res, res, 0, 255, NORM_MINMAX);
    convertScaleAbs(res, res);
    imshow("logTransformImage", res);
    waitKey(0);
    destroyAllWindows();
    return true;
}

/*灰度图像的伽马变换
 *
 * 具体内容：
 *  设置并调整γ值对图像进行伽马变换。
 *
 * */
bool gammaImage(string &src, int flag, double r){
    Mat image, res;
    image = imread(src, flag);
    imshow("inputImage", image);

    //用于伽马变换的图像
    res = image.clone();
    int rows = res.rows;
    int cols = res.cols;

    //伽马变换
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            auto gray = (double)res.at<uchar>(i, j);
            int a = 1;
            gray = a * pow(gray, r);
            res.at<uchar>(i, j) = saturate_cast<uchar>(gray);
        }
    }
    normalize(res, res, 0, 255, NORM_MINMAX);
    convertScaleAbs(res, res);
    imshow("gammaImage", res);
    waitKey(0);
    destroyAllWindows();
    return true;
}

/*彩色图像的补色变换
 *
 * 具体内容：
 *  对彩色图像进行补色变换。
 *
 * */
bool complementColorImage(string &src, int flag){
    Mat image, res;
    image = imread(src, flag);
    imshow("inputImage", image);

    //用于补色变换的图像
    res = image.clone();

    //补色变换
    Vec3b pixel, temp;

    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            pixel = image.at<Vec3b>(i, j);			//RGB->CMY

            /*
			pixel里的三个通道是BGR，其补色是CMY色域的，变换关系如下：
			C=255-R；
			M=255-G；
			Y=255-B；
			*/

            temp[0] = 255 - pixel[2];		//C=255-R；
            temp[1] = 255 - pixel[1];		//M=255-G；
            temp[2] = 255 - pixel[0];		//Y=255-B；

            res.at <Vec3b>(i, j) = temp;

        }
    }
    imshow("complementColorImage", res);
    waitKey(0);
    destroyAllWindows();
    return true;

}
