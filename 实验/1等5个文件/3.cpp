//空域滤波
#include<iostream>
#include<string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>

using namespace std;
using namespace cv;
int MeanFilter_Gray(int a, int b)
{
	Mat image, meanRes;
	image = imread("avatar.jpg", 0); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	blur(image, meanRes, Size(a, b));			//均值滤波

	namedWindow("均值滤波", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("均值滤波", meanRes);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}

int MeanFilter_Color(int a, int b)
{
	Mat image, meanRes;
	image = imread("avatar.jpg", 1); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	blur(image, meanRes, Size(a, b));			//均值滤波

	namedWindow("均值滤波", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("均值滤波", meanRes);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}

int GaussianFilter_Gray(int a, int b)
{
	Mat image, res;
	image = imread("avatar.jpg", 0); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	GaussianBlur(image, res, Size(a, b), 1);

	namedWindow("高斯滤波", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("高斯滤波", res);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}

int GaussianFilter_Color(int a, int b)
{
	Mat image, res;
	image = imread("avatar.jpg", 1); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	GaussianBlur(image, res, Size(a, b), 1);

	namedWindow("高斯滤波", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("高斯滤波", res);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}


int Sobel()
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat image, res;
	image = imread("avatar.jpg", 0); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	Sobel(image, grad_x, image.depth(), 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("【效果图】 X方向Sobel", abs_grad_x);

	Sobel(image, grad_y, image.depth(), 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("【效果图】Y方向Sobel", abs_grad_y);

	//【5】合并梯度(近似)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, res);
	imshow("【效果图】整体方向Sobel", res);

	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}

int Sobel_Color()
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat image, res;
	image = imread("avatar.jpg", 1); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	Sobel(image, grad_x, image.depth(), 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	imshow("【效果图】 X方向Sobel", abs_grad_x);

	Sobel(image, grad_y, image.depth(), 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	imshow("【效果图】Y方向Sobel", abs_grad_y);

	//【5】合并梯度(近似)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, res);
	imshow("【效果图】整体方向Sobel", res);

	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}

//拉普拉斯模板
int Laplacian_Color()
{
	Mat image, res;
	image = imread("avatar.jpg", IMREAD_COLOR); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(image, res, image.depth(), kernel);

	namedWindow("拉普拉斯模板", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("拉普拉斯模板", res);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}

int Laplacian_Gray()
{
	Mat image, res;
	image = imread("avatar.jpg", 0); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(image, res, image.depth(), kernel);

	namedWindow("拉普拉斯模板", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("拉普拉斯模板", res);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}


int Lap2()
{
	Mat image, res;
	image = imread("avatar.jpg", IMREAD_COLOR); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	res.create(image.size(), image.type());//为输出图像分配内容

		/*拉普拉斯滤波核3*3
		 0  -1   0
		-1   5  -1
		 0  -1   0  */
		 //处理除最外围一圈外的所有像素值

	for (int i = 1; i < image.rows - 1; i++)
	{
		const uchar * pre = image.ptr<const uchar>(i - 1);//前一行
		const uchar * cur = image.ptr<const uchar>(i);//当前行，第i行
		const uchar * next = image.ptr<const uchar>(i + 1);//下一行
		uchar * output = res.ptr<uchar>(i);//输出图像的第i行
		int ch = image.channels();//通道个数
		int startCol = ch;//每一行的开始处理点
		int endCol = (image.cols - 1)* ch;//每一行的处理结束点
		for (int j = startCol; j < endCol; j++)
		{
			//输出图像的遍历指针与当前行的指针同步递增, 以每行的每一个像素点的每一个通道值为一个递增量, 因为要

			//考虑到图像的通道数
				//saturate_cast<uchar>保证结果在uchar范围内
			*output++ = saturate_cast<uchar>(5 * cur[j] - pre[j] - next[j] - cur[j - ch] - cur[j + ch]);
		}
	}
	//将最外围一圈的像素值设为0
	res.row(0).setTo(Scalar(0));
	res.row(res.rows - 1).setTo(Scalar(0));
	res.col(0).setTo(Scalar(0));
	res.col(res.cols - 1).setTo(Scalar(0));
	/*/或者也可以尝试将最外围一圈设置为原图的像素值
	image.row(0).copyTo(result.row(0));
	image.row(image.rows-1).copyTo(result.row(result.rows-1));
	image.col(0).copyTo(result.col(0));
	image.col(image.cols-1).copyTo(result.col(result.cols-1));*/
	namedWindow("拉普拉斯模板-手写", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("拉普拉斯模板-手写", res);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}

int Robert_RGB()
{
	Mat image, res;
	image = imread("avatar.jpg", 1); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.

	res = image.clone();

	CvScalar t1, t2, t3, t4, t;
	IplImage res2 = IplImage(image);

	for (int i = 0; i < res2.height - 1; i++)
	{
		for (int j = 0; j < res2.width - 1; j++)
		{
			t1 = cvGet2D(&res2, i, j);
			t2 = cvGet2D(&res2, i + 1, j + 1);
			t3 = cvGet2D(&res2, i, j + 1);
			t4 = cvGet2D(&res2, i + 1, j);


			for (int k = 0; k < 3; k++)
			{
				int t7 = (t1.val[k] - t2.val[k])*(t1.val[k] - t2.val[k]) + (t4.val[k] - t3.val[k])*(t4.val[k] - t3.val[k]);
				t.val[k] = sqrt(t7);
			}
			cvSet2D(&res2, i, j, t);
		}
	}
	res = cvarrToMat(&res2);
	namedWindow("Robert_RGB滤波", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Robert_RGB滤波", res);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;
}

int Robert_G()
{
	Mat image, res;
	image = imread("avatar.jpg", 0); // Read the file
	namedWindow("原图", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("原图", image);                // Show our image inside it.
	res = image.clone();

	for (int i = 0; i < image.rows - 1; i++) {
		for (int j = 0; j < image.cols - 1; j++) {
			//根据公式计算
			int t1 = (image.at<uchar>(i, j) -
				image.at<uchar>(i + 1, j + 1))*
				(image.at<uchar>(i, j) -
					image.at<uchar>(i + 1, j + 1));
			int t2 = (image.at<uchar>(i + 1, j) -
				image.at<uchar>(i, j + 1))*
				(image.at<uchar>(i + 1, j) -
					image.at<uchar>(i, j + 1));
			//计算g（x,y）
			res.at<uchar>(i, j) = (uchar)sqrt(t1 + t2);
		}
	}

	namedWindow("Robert_G滤波", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Robert_G滤波", res);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
	return 0;

}


void EnhanceFilter(Mat img, Mat &dst, double dProportion, int nTempH, int nTempW, int nTempMY, int nTempMX, float *pfArray, float fCoef)
{


	int i, j, nHeight = img.rows, nWidth = img.cols;
	vector<vector<int>> GrayMat1, GrayMat2, GrayMat3;//暂存按比例叠加图像，R,G,B三通道
	vector<int> vecRow1(nWidth, 0), vecRow2(nWidth, 0), vecRow3(nWidth, 0);
	for (i = 0; i < nHeight; i++)
	{
		GrayMat1.push_back(vecRow1);
		GrayMat2.push_back(vecRow2);
		GrayMat3.push_back(vecRow3);
	}

	//锐化图像，输出带符号响应，并与原图像按比例叠加
	for (i = nTempMY; i < nHeight - (nTempH - nTempMY) + 1; i++)
	{
		for (j = nTempMX; j < nWidth - (nTempW - nTempMX) + 1; j++)
		{
			float fResult1 = 0;
			float fResult2 = 0;
			float fResult3 = 0;
			for (int k = 0; k < nTempH; k++)
			{
				for (int l = 0; l < nTempW; l++)
				{
					//分别计算三通道加权和
					fResult1 += img.at<Vec3b>(i, j)[0] * pfArray[k*nTempW + 1];
					fResult2 += img.at<Vec3b>(i, j)[1] * pfArray[k*nTempW + 1];
					fResult3 += img.at<Vec3b>(i, j)[2] * pfArray[k*nTempW + 1];
				}
			}

			//三通道加权和分别乘以系数并限制响应范围，最后和原图像按比例混合
			fResult1 *= fCoef;
			if (fResult1 > 255)
				fResult1 = 255;
			if (fResult1 < -255)
				fResult1 = -255;
			GrayMat1[i][j] = dProportion * img.at<Vec3b>(i, j)[0] + fResult1 + 0.5;

			fResult2 *= fCoef;
			if (fResult2 > 255)
				fResult2 = 255;
			if (fResult2 < -255)
				fResult2 = -255;
			GrayMat2[i][j] = dProportion * img.at<Vec3b>(i, j)[1] + fResult2 + 0.5;

			fResult3 *= fCoef;
			if (fResult3 > 255)
				fResult3 = 255;
			if (fResult3 < -255)
				fResult3 = -255;
			GrayMat3[i][j] = dProportion * img.at<Vec3b>(i, j)[2] + fResult3 + 0.5;
		}
	}
	int nMax1 = 0, nMax2 = 0, nMax3 = 0;//三通道最大灰度和值
	int nMin1 = 65535, nMin2 = 65535, nMin3 = 65535;//三通道最小灰度和值
	//分别统计三通道最大值最小值
	for (i = nTempMY; i < nHeight - (nTempH - nTempMY) + 1; i++)
	{
		for (j = nTempMX; j < nWidth - (nTempW - nTempMX) + 1; j++)
		{
			if (GrayMat1[i][j] > nMax1)
				nMax1 = GrayMat1[i][j];
			if (GrayMat1[i][j] < nMin1)
				nMin1 = GrayMat1[i][j];

			if (GrayMat2[i][j] > nMax2)
				nMax2 = GrayMat2[i][j];
			if (GrayMat2[i][j] < nMin2)
				nMin2 = GrayMat2[i][j];

			if (GrayMat3[i][j] > nMax3)
				nMax3 = GrayMat3[i][j];
			if (GrayMat3[i][j] < nMin3)
				nMin3 = GrayMat3[i][j];
		}
	}
	//将按比例叠加后的三通道图像取值范围重新归一化到[0,255]
	int nSpan1 = nMax1 - nMin1, nSpan2 = nMax2 - nMin2, nSpan3 = nMax3 - nMin3;
	for (i = nTempMY; i < nHeight - (nTempH - nTempMY) + 1; i++)
	{
		for (j = nTempMX; j < nWidth - (nTempW - nTempMX) + 1; j++)
		{
			int br, bg, bb;
			if (nSpan1 > 0)
				br = (GrayMat1[i][j] - nMin1) * 255 / nSpan1;
			else if (GrayMat1[i][j] <= 255)
				br = GrayMat1[i][j];
			else
				br = 255;
			dst.at<Vec3b>(i, j)[0] = br;

			if (nSpan2 > 0)
				bg = (GrayMat2[i][j] - nMin2) * 255 / nSpan2;
			else if (GrayMat2[i][j] <= 255)
				bg = GrayMat2[i][j];
			else
				bg = 255;
			dst.at<Vec3b>(i, j)[1] = bg;

			if (nSpan3 > 0)
				bb = (GrayMat3[i][j] - nMin3) * 255 / nSpan3;
			else if (GrayMat3[i][j] <= 255)
				bb = GrayMat3[i][j];
			else
				bb = 255;
			dst.at<Vec3b>(i, j)[2] = bb;
		}
	}
}

void test4()
{
	Mat img = imread("avatar.jpg");
	imshow("原图", img);
	Mat dst = img.clone();
	//常用滤波模板数组
//平均平滑1/9
	float Template_Smooth_Avg[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	//Gauss平滑1/16
	float Template_Smooth_Gauss[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	//Sobel垂直边缘检测
	float Template_Smooth_HSobel[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	//Sobel水平边缘检测
	float Template_Smooth_VSobel[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	//LOG边缘检测
	float Template_Log[25] = { 0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0 };
	//Laplacian边缘检测
	float Template_Laplacian1[9] = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };//对90度各向同性
	float Template_Laplacian2[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };//对45度各向同性
	/*************************************************************************************************************
	高提升滤波
	dProportion：高提升滤波中原图像的混合比例
	nTempH：模板高度，nTempW：模板宽度
	nTempMY：模板中心元素坐标，nTempMX：模板中心元素坐标
	fpArray：指向模板数组的指针，可以选取不同模板实现不同滤波的高提升版本
	fCoef：模板系数
	**************************************************************************************************************/
	EnhanceFilter(img, dst, 1.8, 3, 3, 1, 1, Template_Laplacian2, 1);

	imshow("高提升Laplacian", dst);
	waitKey(0);
	destroyAllWindows();
	return;
}


int main()
{

	MeanFilter_Gray(3, 3);
	MeanFilter_Gray(5, 5);
	MeanFilter_Gray(9, 9);
	GaussianFilter_Gray(3, 3);
	GaussianFilter_Gray(5, 5);
	GaussianFilter_Gray(9, 9);
	Laplacian_Gray();
	Robert_G();
	Sobel();
	test4();
	MeanFilter_Color(3, 3);
	MeanFilter_Color(5, 5);
	MeanFilter_Color(9, 9);
	GaussianFilter_Color(3, 3);
	GaussianFilter_Color(5, 5);
	GaussianFilter_Color(9, 9);
	Lap2();
	Laplacian_Color();
	Robert_RGB();
	Sobel_Color();
	return 0;
}