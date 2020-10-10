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

void gammaTransform(string &, double r = 0.6);

int main(){
	string str = "G://1.jpg";
	gammaTransform(str);
}

/*灰度图像的伽马变换
 *
 * 具体内容：
 *  设置并调整γ值对图像进行伽马变换。
 *
 * */
void gammaTransform(string &src, double r) {
	Mat image, res;
	image = imread(src, 0);
	imshow("初始图片", image);

	//用于伽马变换的图像
	res = image.clone();
	int rows = res.rows;
	int cols = res.cols;

	//伽马变换
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//将uchar灰度值转化为double值，方便进行计算
			auto gray = (double)res.at<uchar>(i, j);
			int a = 1;
			//伽马变换，r为伽马值
			gray = a * pow(gray, r);
			//输出限制范围
			res.at<uchar>(i, j) = saturate_cast<uchar>(gray);
		}
	}
	//对double变量进行归一化
	normalize(res, res, 0, 255, NORM_MINMAX);
	//double型转换为uchar
	convertScaleAbs(res, res);
	imshow("伽马变换后的图像", res);
	waitKey(0);
	destroyAllWindows();
}