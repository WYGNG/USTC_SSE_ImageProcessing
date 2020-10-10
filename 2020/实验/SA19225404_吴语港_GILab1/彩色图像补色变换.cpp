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

void complementTransform(string &);

int main(){
	string str = "G://1.jpg";
	complementTransform(str);
}

/*彩色图像的补色变换
 *
 * 具体内容：
 *  对彩色图像进行补色变换。
 *
 * */
void complementTransform(string &src) {
	Mat image, res;
	//flag标志位为1，表示读入彩色图像
	image = imread(src, 1);
	imshow("初始图片", image);

	//用于补色变换的图像
	res = image.clone();

	//补色变换对像素点处理的临时变量
	//vec3b类型为3个通道的uchar，更好对应rgb图单像素的3个通道值
	Vec3b pixel, temp;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			/*
			RGB->CMY
			pixel里的三个通道是BGR，其补色是CMY色域的，变换关系如下：
			C=255-R；
			M=255-G；
			Y=255-B；
			*/
			//注意3个通道中的下标[0],[1],[2]分别表示BGR
			temp[0] = 255 - image.at<Vec3b>(i, j)[2];		//C=255-R；
			temp[1] = 255 - image.at<Vec3b>(i, j)[1];		//M=255-G；
			temp[2] = 255 - image.at<Vec3b>(i, j)[0];		//Y=255-B；

			res.at <Vec3b>(i, j) = temp;

		}
	}
	imshow("补色变换后的图像", res);
	waitKey(0);
	destroyAllWindows();
}
