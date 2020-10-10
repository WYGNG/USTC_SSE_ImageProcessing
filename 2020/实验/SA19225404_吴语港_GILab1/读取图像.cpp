#include<iostream>
#include<string>

#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>

using namespace std;
using namespace cv;

//函数声明，可以不带具体参数名， 编译器知道类型就可以了
void ReadImage(string &, int32_t); 


int main() {
	string str = "G:\\1.jpg";
	ReadImage(str, 1);
}

/* 利用 OpenCV 读取图像
 *
 * 具体内容
 *  用打开 OpenCV 打开图像，并在窗口中显示
 *
 * */
void ReadImage(string &src, int flag = 1) {
    //读取图片
	Mat img = imread(src, flag);
    //显示图片
	imshow("ShowImage", img);
    //等待按键，保持图片
	waitKey(0);
    //关闭所有窗口
	destroyAllWindows();
}