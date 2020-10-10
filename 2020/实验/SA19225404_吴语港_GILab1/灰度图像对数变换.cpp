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

void logTransform(string &, double r = 1);

int main(){
	string str = "G://1.jpg";
	logTransform(str);
}

/*灰度图像的对数变换
 *
 * 具体内容：
 *  设置并调整 r 值对图像进行对数变换。
 *
 * */
void logTransform(string &src, double r) {
	Mat image, res;
	image = imread(src, 0);
	imshow("初始图片", image);

	//用于返回对数变换结果的mat型变量res
	res = image.clone();
	int rows = res.rows;
	int cols = res.cols;

	//对数变换 
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//gray为double型变量，将像素点的灰度值强制转换为double型，方便进行对数运算
			auto gray = (double)res.at<uchar>(i, j);
			//进行对数变换
			gray = r * log((double)(1 + gray));
			//限制输入输出范围
			res.at<uchar>(i, j) = saturate_cast<uchar>(gray);
		}
	}
	//对对数变换后的值进行归一化处理
	normalize(res, res, 0, 255, NORM_MINMAX);
	convertScaleAbs(res, res);
	imshow("逻辑变换后的图像", res);
	waitKey(0);
	destroyAllWindows();
}