//图像灰度变换
#include<iostream>
#include<string>
#include<vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2\opencv.hpp>   
#include<opencv2\highgui\highgui.hpp>
using namespace std;
using namespace cv;

int swell()
{
	Mat img = imread("lena.jpg");
	imshow("原始图", img);
	Mat out;
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
	//膨胀操作
	dilate(img, out, element);
	imshow("膨胀操作", out);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

void corrosion()
{
	Mat img = imread("lena.jpg");
	imshow("原始图", img);
	Mat out;
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
	//腐蚀操作
	erode(img, out, element);
	imshow("腐蚀操作", out);
	waitKey(0);
	destroyAllWindows();
	return ;
}

void morphology()
{
	vector<int> v = { MORPH_OPEN, MORPH_CLOSE,MORPH_GRADIENT,MORPH_TOPHAT,MORPH_BLACKHAT,MORPH_ERODE,MORPH_DILATE};
	//开运算，闭运算，形态学梯度，顶帽，黑帽，腐蚀，膨胀。
	Mat img = imread("lena.jpg");
	imshow("原始图", img);
	Mat out;
	//获取自定义核
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的

	//高级形态学处理，调用这个函数就可以了，具体要选择哪种操作，就修改第三个参数就可以了。这里演示的是形态学梯度处理
	morphologyEx(img, out, MORPH_GRADIENT, element);
	imshow("形态学处理操作", out);
	waitKey(0);
	destroyAllWindows();
	return;
}
int main()
{
	swell();
	corrosion();
	return 0;
}