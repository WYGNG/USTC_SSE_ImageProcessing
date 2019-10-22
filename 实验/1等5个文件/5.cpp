//图像灰度变换
#include<iostream>
#include<string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


using namespace std;
using namespace cv;

Mat DFT(Mat I)
{
	//Mat I = imread("avatar.jpg", IMREAD_GRAYSCALE);       //读入图像灰度图

	Mat padded;                 //以0填充输入图像矩阵
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols);

	//填充输入图像I，输入矩阵为padded，上方和左方不做填充处理
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);     //将planes融合合并成一个多通道数组complexI

	dft(complexI, complexI);        //进行傅里叶变换

	//计算幅值，转换到对数尺度(logarithmic scale)
	//=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);        //planes[0] = Re(DFT(I),planes[1] = Im(DFT(I))
									//即planes[0]为实部,planes[1]为虚部
	magnitude(planes[0], planes[1], planes[0]);     //planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);
	log(magI, magI);                //转换到对数尺度(logarithmic scale)

	//如果有奇数行或列，则对频谱进行裁剪
	magI = magI(Rect(0, 0, magI.cols&-2, magI.rows&-2));

	//重新排列傅里叶图像中的象限，使得原点位于图像中心
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));       //左上角图像划定ROI区域
	Mat q1(magI, Rect(cx, 0, cx, cy));      //右上角图像
	Mat q2(magI, Rect(0, cy, cx, cy));      //左下角图像
	Mat q3(magI, Rect(cx, cy, cx, cy));     //右下角图像

	//变换左上角和右下角象限
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	//变换右上角和左下角象限
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式
	normalize(magI, magI, 0, 1, 32);

	return magI;
}

void test7()
{
	Mat image, res;
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);

	res = DFT(image);
	imshow("频谱图", res);
	waitKey(0);
	destroyAllWindows();
	return;
}

int DFTAndIDFT()
{
	Mat input = imread("avatar.jpg", 0);
	imshow("input", input);//显示原图
	int w = getOptimalDFTSize(input.cols);
	int h = getOptimalDFTSize(input.rows);//获取最佳尺寸，快速傅立叶变换要求尺寸为2的n次方
	Mat padded;
	copyMakeBorder(input, padded, 0, h - input.rows, 0, w - input.cols, BORDER_CONSTANT, Scalar::all(0));//填充图像保存到padded中
	Mat plane[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };//创建通道
	Mat complexIm;
	merge(plane, 2, complexIm);//合并通道
	dft(complexIm, complexIm);//进行傅立叶变换，结果保存在自身
	split(complexIm, plane);//分离通道
	magnitude(plane[0], plane[1], plane[0]);//获取幅度图像，0通道为实数通道，1为虚数，因为二维傅立叶变换结果是复数
	int cx = padded.cols / 2; int cy = padded.rows / 2;//一下的操作是移动图像，左上与右下交换位置，右上与左下交换位置
	Mat temp;
	Mat part1(plane[0], Rect(0, 0, cx, cy));
	Mat part2(plane[0], Rect(cx, 0, cx, cy));
	Mat part3(plane[0], Rect(0, cy, cx, cy));
	Mat part4(plane[0], Rect(cx, cy, cx, cy));
	part1.copyTo(temp);
	part4.copyTo(part1);
	temp.copyTo(part4);
	part2.copyTo(temp);
	part3.copyTo(part2);
	temp.copyTo(part3);
	//*******************************************************************

	Mat _complexim;
	complexIm.copyTo(_complexim);//把变换结果复制一份，进行逆变换，也就是恢复原图
	Mat iDft[] = { Mat::zeros(plane[0].size(),CV_32F),Mat::zeros(plane[0].size(),CV_32F) };//创建两个通道，类型为float，大小为填充后的尺寸
	idft(_complexim, _complexim);//傅立叶逆变换
	split(_complexim, iDft);//结果貌似也是复数
	magnitude(iDft[0], iDft[1], iDft[0]);//分离通道，主要获取0通道
	normalize(iDft[0], iDft[0], 1, 0, CV_MINMAX);//归一化处理，float类型的显示范围为0-1,大于1为白色，小于0为黑色
	imshow("idft", iDft[0]);//显示逆变换
	//*******************************************************************
	plane[0] += Scalar::all(1);//傅立叶变换后的图片不好分析，进行对数处理，结果比较好看
	log(plane[0], plane[0]);
	normalize(plane[0], plane[0], 1, 0, CV_MINMAX);
	imshow("dft", plane[0]);
	waitKey(0);
	destroyAllWindows();
	return 0;
}

void ideal_Low_Pass_Filter(double D0 = 60)
{
	Mat src, fourier, res;
	src = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", src);
	Mat img = src.clone();
	//cvtColor(src, img, CV_BGR2GRAY);
	//调整图像加速傅里叶变换
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	//记录傅里叶变换的实部和虚部
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	//进行傅里叶变换
	dft(complexImg, complexImg);
	//获取图像
	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));//这里为什么&上-2具体查看opencv文档
	//其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0
	//获取中心点坐标
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	//调整频域
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//Do为自己设定的阀值具体看公式

	//处理按公式保留中心部分
	for (int y = 0; y < mag.rows; y++) {
		double* data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++) {
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			if (d <= D0)
			{

			}
			else {
				data[x] = 0;
			}
		}
	}
	//再调整频域
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//逆变换
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("理想低通滤波器", invDFTcvt);
	waitKey(0);
	destroyAllWindows();
	return;
}

void ideal_High_Pass_Filter(double D0 = 60)
{
	Mat src, fourier, res;
	src = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", src);
	Mat img = src.clone();
	//cvtColor(src, img, CV_BGR2GRAY);
	//调整图像加速傅里叶变换
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));
	//记录傅里叶变换的实部和虚部
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	//进行傅里叶变换
	dft(complexImg, complexImg);
	//获取图像
	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));//这里为什么&上-2具体查看opencv文档
	//其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0
	//获取中心点坐标
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	//调整频域
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//Do为自己设定的阀值具体看公式
	//处理按公式保留中心部分
	for (int y = 0; y < mag.rows; y++) {
		double* data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++) {
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			if (d <= D0)
			{
				data[x] = 0;
			}
			else
			{
				/*和低通相反，通过高的，所以这里啥都没有*/
			}
		}
	}
	//再调整频域
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//逆变换
	Mat invDFT, invDFTcvt;
	idft(mag, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("理想高通滤波器", invDFTcvt);
	waitKey(0);
	destroyAllWindows();
	return;
}

void Butterworth_Low_Paass_Filter(double D0 = 60, int n = 2)
{
	Mat src, fourier, res;
	src = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", src);

	//H = 1 / (1+(D/D0)^2n)
	Mat img = src.clone();
	//cvtColor(src, img, CV_BGR2GRAY);
	//调整图像加速傅里叶变换
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);



	for (int y = 0; y < mag.rows; y++)
	{
		double* data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++)
		{
			//cout << data[x] << endl;
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			//cout << d << endl;
			double h = 1.0 / (1 + pow(d / D0, 2 * n));
			if (h <= 0.5)
			{
				data[x] = 0;
			}
			else {
				//data[x] = data[x]*0.5;
				//cout << h << endl;
			}

			//cout << data[x] << endl;
		}
	}
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//逆变换
	Mat invDFT, invDFTcvt;
	idft(complexImg, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("巴特沃斯低通滤波器", invDFTcvt);

	waitKey(0);
	destroyAllWindows();
	return;
}

void Butterworth_High_Paass_Filter(double D0 = 60, int n = 2)
{
	Mat src, fourier, res;
	src = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", src);

	//H = 1 / (1+(D/D0)^2n)
	Mat img = src.clone();
	//cvtColor(src, img, CV_BGR2GRAY);
	//调整图像加速傅里叶变换
	int M = getOptimalDFTSize(img.rows);
	int N = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, M - img.rows, 0, N - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	Mat mag = complexImg;
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));

	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	for (int y = 0; y < mag.rows; y++)
	{
		double* data = mag.ptr<double>(y);
		for (int x = 0; x < mag.cols; x++)
		{
			//cout << data[x] << endl;
			double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
			//cout << d << endl;
			double h = 1.0 / (1 + pow(D0 / d, 2 * n));
			if (h <= 0.5)
			{
				data[x] = 0;
			}
			else {
				//data[x] = data[x]*0.5;
				//cout << h << endl;
			}

			//cout << data[x] << endl;
		}
	}
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	//逆变换
	Mat invDFT, invDFTcvt;
	idft(complexImg, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("巴特沃斯高通滤波器", invDFTcvt);

	waitKey(0);
	destroyAllWindows();
	return;
}


int main()
{
	DFTAndIDFT();
	ideal_Low_Pass_Filter(40.0);
	ideal_High_Pass_Filter(40.0);
	Butterworth_Low_Paass_Filter(40, 2);
	Butterworth_High_Paass_Filter(40, 2);

	return 0;
}