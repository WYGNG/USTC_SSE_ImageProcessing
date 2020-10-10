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

//默认的灰度阈值设置为140
void toBinaryImage(string &, int r = 180);

int main(){
	string str = "G://1.jpg";
	toBinaryImage(str);
}

/*灰度图像二值化处理
 *
 * 具体内容
 *  设置并调整阈值对图像进行二值化处理。
 *
 * */
void toBinaryImage(string &src, int r) {
	Mat image, res;
	//imread函数的参数1为图片的路径，参数2为图片的读取方式
	//0为读取为灰度图， 1为读取为彩色图
	image = imread(src, 0);
	imshow("初始图片", image);

	//用于二值化的图像res
	res = image.clone();
	int rows = res.rows;
	int cols = res.cols;

	//二值化
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			//res为mat型变量，用了at函数对图片上固定位置的像素点进行操作
			auto gray = res.at<uchar>(i, j);
			//根据阈值对像素点的灰度值划分，二值化
			if (gray > r)   gray = 255;	//纯白色
			else    gray = 0;			//纯黑色
			//saturate_cast函数的作用即是：当运算完之后，结果为负，则转为0，结果超出255，则为255
			res.at<uchar>(i, j) = saturate_cast<uchar>(gray);
		}
	}
	imshow("二值化后的图片", res);
	waitKey(0);
	destroyAllWindows();
}