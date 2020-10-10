//图像灰度变换
#include <iostream>
#include <string>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <windows.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
//#include <opencv2/core/hal/intrin_sse.hpp>
using namespace std;
using namespace cv;


//本函数加入盐噪声
void addSalt(Mat& image, int n)
{
	srand((unsigned)time(NULL));
	int i, j;
	for (int k = 0; k < n; k++)//将图像中n个像素随机置零
	{
		i = rand() % image.cols;
		j = rand() % image.rows;
		//将图像颜色随机改变
		if (image.channels() == 1)
			image.at<uchar>(j, i) = 255;
		else
		{
			for (int t = 0; t < image.channels(); t++)
			{
				image.at<Vec3b>(j, i)[t] = 255;
			}
		}
	}
}

void addPepper(Mat& image, int n)//本函数加入椒噪声
{
	srand((unsigned)time(NULL));
	for (int k = 0; k < n; k++)//将图像中n个像素随机置零
	{
		int i = rand() % image.cols;
		int j = rand() % image.rows;
		//将图像颜色随机改变
		if (image.channels() == 1)
			image.at<uchar>(j, i) = 0;
		else
		{
			for (int t = 0; t < image.channels(); t++)
			{
				image.at<Vec3b>(j, i)[t] = 0;
			}
		}

	}
}

int GaussianNoise(double mu, double sigma)
{
	//定义一个特别小的值
	const double epsilon = numeric_limits<double>::min();//返回目标数据类型能表示的最逼近1的正数和1的差的绝对值
	static double z0, z1;
	static bool flag = false;
	flag = !flag;
	//flag为假，构造高斯随机变量
	if (!flag)
		return z1 * sigma + mu;
	double u1, u2;
	//构造随机变量

	do
	{
		u1 = rand()*(1.0 / RAND_MAX);
		u2 = rand()*(1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	//flag为真构造高斯随机变量X
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI * u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI * u2);
	return z1 * sigma + mu;
}

Mat addGaussianNoise(Mat& srcImage)
{
	Mat resultImage = srcImage.clone();    //深拷贝,克隆
	int channels = resultImage.channels();    //获取图像的通道
	int nRows = resultImage.rows;    //图像的行数

	int nCols = resultImage.cols*channels;   //图像的总列数
	//判断图像的连续性
	if (resultImage.isContinuous())    //判断矩阵是否连续，若连续，我们相当于只需要遍历一个一维数组 
	{
		nCols *= nRows;
		nRows = 1;
	}
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{	//添加高斯噪声
			int val = resultImage.ptr<uchar>(i)[j] + GaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			resultImage.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return resultImage;
}

//中值滤波器
void medeanFilter(Mat& src, int win_size) {
	int rows = src.rows, cols = src.cols;
	int start = win_size / 2;
	for (int m = start; m < rows - start; m++) {
		for (int n = start; n < cols - start; n++) {
			vector<uchar> model;
			for (int i = -start + m; i <= start + m; i++) {
				for (int j = -start + n; j <= start + n; j++) {
					//cout << int(src.at<uchar>(i, j)) << endl;
					model.push_back(src.at<uchar>(i, j));
				}
			}
			sort(model.begin(), model.end());     //采用快速排序进行
			src.at<uchar>(m, n) = model[win_size*win_size / 2];
		}
	}
}

//算术均值滤波器
void meanFilter(Mat& src, int win_size) {
	int rows = src.rows, cols = src.cols;
	int start = win_size / 2;
	for (int m = start; m < rows - start; m++) {
		for (int n = start; n < cols - start; n++) {
			if (src.channels() == 1)				//灰色图
			{
				int sum = 0;
				for (int i = -start + m; i <= start + m; i++)
				{
					for (int j = -start + n; j <= start + n; j++) {
						sum += src.at<uchar>(i, j);
					}
				}
				src.at<uchar>(m, n) = uchar(sum / win_size / win_size);
			}
			else
			{
				Vec3b pixel;
				int sum1[3] = { 0 };
				for (int i = -start + m; i <= start + m; i++)
				{
					for (int j = -start + n; j <= start + n; j++)
					{
						pixel = src.at<Vec3b>(i, j);
						for (int k = 0; k < src.channels(); k++)
						{
							sum1[k] += pixel[k];
						}
					}

				}
				for (int k = 0; k < src.channels(); k++)
				{
					pixel[k] = sum1[k] / win_size / win_size;
				}
				src.at<Vec3b>(m, n) = pixel;
			}
		}
	}
}

//几何均值滤波器
Mat GeometryMeanFilter(Mat src)
{
	Mat dst = src.clone();
	int row, col;
	int h = src.rows;
	int w = src.cols;
	double mul;
	double dc;
	int mn;
	//计算每个像素的去噪后 color 值
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{

			if (src.channels() == 1)				//灰色图
			{
				mul = 1.0;
				mn = 0;
				//统计邻域内的几何平均值，邻域大小 5*5
				for (int m = -2; m <= 2; m++) {
					row = i + m;
					for (int n = -2; n <= 2; n++) {
						col = j + n;
						if (row >= 0 && row < h && col >= 0 && col < w) {
							int s = src.at<uchar>(row, col);
							mul = mul * (s == 0 ? 1 : s); //邻域内的非零像素点相乘，最小值设定为1
							mn++;
						}
					}
				}
				//计算 1/mn 次方
				dc = pow(mul, 1.0 / mn);
				//统计成功赋给去噪后图像。
				int res = (int)dc;
				dst.at<uchar>(i, j) = res;
			}
			else
			{
				double multi[3] = { 1.0,1.0,1.0 };
				mn = 0;
				Vec3b pixel;

				for (int m = -2; m <= 2; m++)
				{
					row = i + m;
					for (int n = -2; n <= 2; n++)
					{
						col = j + n;
						if (row >= 0 && row < h && col >= 0 && col < w)
						{
							pixel = src.at<Vec3b>(row, col);
							for (int k = 0; k < src.channels(); k++)
							{
								multi[k] = multi[k] * (pixel[k] == 0 ? 1 : pixel[k]);//邻域内的非零像素点相乘，最小值设定为1
							}
							mn++;
						}
					}
				}
				double d;
				for (int k = 0; k < src.channels(); k++)
				{
					d = pow(multi[k], 1.0 / mn);
					pixel[k] = (int)d;
				}
				dst.at<Vec3b>(i, j) = pixel;
			}
		}
	}
	return dst;
}

//谐波均值滤波器——模板大小 5*5
Mat HarmonicMeanFilter(Mat src)
{
	//IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	Mat dst = src.clone();
	int row, col;
	int h = src.rows;
	int w = src.cols;
	double sum;
	double dc;
	int mn;
	//计算每个像素的去噪后 color 值
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			sum = 0.0;
			mn = 0;
			//统计邻域,5*5 模板
			for (int m = -2; m <= 2; m++) {
				row = i + m;
				for (int n = -2; n <= 2; n++) {
					col = j + n;
					if (row >= 0 && row < h && col >= 0 && col < w) {
						int s = src.at<uchar>(row, col);
						sum = sum + (s == 0 ? 255 : 255.0 / s);					//如果是0，设定为255
						mn++;
					}
				}
			}
			int d;
			dc = mn * 255.0 / sum;
			d = dc;
			//统计成功赋给去噪后图像。
			dst.at<uchar>(i, j) = d;
		}
	}
	return dst;
}

//逆谐波均值大小滤波器——模板大小 5*5
Mat InverseHarmonicMeanFilter(Mat src, double Q)
{
	Mat dst = src.clone();
	int row, col;
	int h = src.rows;
	int w = src.cols;
	double sum;
	double sum1;
	double dc;
	//double Q = 2;
	//计算每个像素的去噪后 color 值
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			sum = 0.0;
			sum1 = 0.0;
			//统计邻域
			for (int m = -2; m <= 2; m++) {
				row = i + m;
				for (int n = -2; n <= 2; n++) {
					col = j + n;
					if (row >= 0 && row < h && col >= 0 && col < w) {

						int s = src.at<uchar>(row, col);
						sum = sum + pow(s, Q + 1);
						sum1 = sum1 + pow(s, Q);
					}
				}
			}
			//计算 1/mn 次方
			int d;
			dc = sum1 == 0 ? 0 : (sum / sum1);
			d = (int)dc;
			//统计成功赋给去噪后图像。
			dst.at<uchar>(i, j) = d;
		}
	}
	return dst;
}

//自适应中值滤波
Mat SelfAdaptMedianFilter(Mat src)
{
	Mat dst = src.clone();
	int row, col;
	int h = src.rows;
	int w = src.cols;
	double Zmin, Zmax, Zmed, Zxy, Smax = 7;
	int wsize;
	//计算每个像素的去噪后 color 值
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			//统计邻域
			wsize = 1;
			while (wsize <= 3) {
				Zmin = 255.0;
				Zmax = 0.0;
				Zmed = 0.0;
				int  Zxy = src.at<uchar>(i, j);
				int mn = 0;
				for (int m = -wsize; m <= wsize; m++)
				{
					row = i + m;
					for (int n = -wsize; n <= wsize; n++)
					{
						col = j + n;
						if (row >= 0 && row < h && col >= 0 && col < w)
						{
							int s = src.at<uchar>(row, col);
							if (s > Zmax)
							{
								Zmax = s;
							}
							if (s < Zmin)
							{
								Zmin = s;
							}
							Zmed = Zmed + s;
							mn++;
						}
					}
				}
				Zmed = Zmed / mn;
				int d;
				if ((Zmed - Zmin) > 0 && (Zmed - Zmax) < 0) {
					if ((Zxy - Zmin) > 0 && (Zxy - Zmax) < 0) {
						d = Zxy;
					}
					else {
						d = Zmed;
					}
					dst.at<uchar>(i, j) = d;
					break;
				}
				else {
					wsize++;
					if (wsize > 3) {
						int d;
						d = Zmed;
						dst.at<uchar>(i, j) = d;
						break;
					}
				}
			}
		}
	}
	return dst;
}

//自适应均值滤波
Mat SelfAdaptMeanFilter(Mat src)
{
	Mat dst = src.clone();
	blur(src, dst, Size(7, 7));
	int row, col;
	int h = src.rows;
	int w = src.cols;
	int mn;
	double Zxy;
	double Zmed;
	double Sxy;
	double Sl;
	double Sn = 100;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			int Zxy = src.at<uchar>(i, j);
			int Zmed = src.at<uchar>(i, j);
			Sl = 0;
			mn = 0;
			for (int m = -3; m <= 3; m++) {
				row = i + m;
				for (int n = -3; n <= 3; n++) {
					col = j + n;
					if (row >= 0 && row < h && col >= 0 && col < w) {
						int Sxy = src.at<uchar>(row, col);
						Sl = Sl + pow(Sxy - Zmed, 2);
						mn++;
					}
				}
			}
			Sl = Sl / mn;
			int d = (int)(Zxy - Sn / Sl * (Zxy - Zmed));
			dst.at<uchar>(i, j) = d;
		}
	}
	return dst;
}

IplImage * MatToIplImage(Mat image)
{
	Mat t = image.clone();
	IplImage *res = &IplImage(t);
	return res;
}

Mat IplImageToMat(IplImage* image)
{
	Mat res = cvarrToMat(image, true);
	return res;
}

void test1()
{
	Mat image, noise, res;

	/*----------高斯噪声+算术均值-----------*/
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);                // Show our image inside it.

	noise = addGaussianNoise(image);			//添加高斯噪声
	imshow("添加高斯噪声", noise);

	res = noise.clone();
	meanFilter(res, 5);					//算术均值滤波器
	imshow("算术均值滤波器", res);
	/*------展示图像-------*/
	waitKey(0);
	destroyAllWindows();

	/*----------胡椒噪声+几何均值-----------*/
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);                // Show our image inside it.

	noise = image.clone();
	addPepper(noise, 1000);
	imshow("添加1000个胡椒噪声", noise);

	res = noise.clone();
	meanFilter(res, 5);
	imshow("几何均值滤波器", res);

	/*------展示图像-------*/
	waitKey(0);
	destroyAllWindows();

	/*--------------盐噪声+谐波均值滤波器------------*/
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);                // Show our image inside it.

	noise = image.clone();
	addSalt(noise, 1000);
	imshow("添加1000个盐噪声", noise);

	res = HarmonicMeanFilter(noise);
	imshow("5*5谐波均值滤波器", res);

	/*------展示图像-------*/
	waitKey(0);
	destroyAllWindows();


	/*-----------椒盐噪声+逆谐波均值滤波器-----------*/
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);                // Show our image inside it.

	noise = image.clone();
	addSalt(noise, 1000);
	Sleep(2000);					//防止随机数种子一样
	addPepper(noise, 1000);
	imshow("添加1000个盐噪声+1000个胡椒噪声", noise);

	res = InverseHarmonicMeanFilter(noise, 1);				//第二个参数是Q，Q=0退化成算术均值
	imshow("5*5逆谐波均值滤波器", res);

	/*------展示图像-------*/
	waitKey(0);
	destroyAllWindows();
	return;
}

void test2()
{
	Mat image, noise, res1, res2;

	/*---------胡椒------------*/
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);                // Show our image inside it.

	noise = image.clone();
	addPepper(noise, 1000);
	imshow("添加1000个胡椒噪声", noise);

	res1 = noise.clone();
	medeanFilter(res1, 5);
	imshow("5*5中均值滤波器", res1);

	res2 = noise.clone();

	res2 = noise.clone();
	medeanFilter(res2, 9);
	imshow("9*9中均值滤波器", res2);
	/*------展示图像-------*/
	waitKey(0);
	destroyAllWindows();

	/*-----------盐噪声---------------*/
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);                // Show our image inside it.

	noise = image.clone();
	addSalt(noise, 1000);
	imshow("添加1000个盐噪声", noise);

	res1 = noise.clone();
	medeanFilter(res1, 5);
	imshow("5*5中均值滤波器", res1);

	res2 = noise.clone();

	res2 = noise.clone();
	medeanFilter(res2, 9);
	imshow("9*9中均值滤波器", res2);
	/*------展示图像-------*/
	waitKey(0);
	destroyAllWindows();


	/*-----------盐噪声+胡椒噪声---------------*/
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);                // Show our image inside it.

	noise = image.clone();
	addSalt(noise, 1000);
	addPepper(noise, 1000);
	imshow("添加1000个盐噪声+1000个胡椒噪声", noise);

	res1 = noise.clone();
	medeanFilter(res1, 5);
	imshow("5*5中均值滤波器", res1);

	res2 = noise.clone();

	res2 = noise.clone();
	medeanFilter(res2, 9);
	imshow("9*9中均值滤波器", res2);
	/*------展示图像-------*/
	waitKey(0);
	destroyAllWindows();
	return;
}

void test3()
{
	Mat image, res1, res2, noise;
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);

	noise = image.clone();
	addPepper(noise, 1000);
	Sleep(2000);
	addSalt(noise, 1000);
	imshow("添加1000个胡椒噪声+1000个盐噪声", noise);

	res1 = SelfAdaptMeanFilter(image);
	imshow("自适应均值滤波", res1);

	res2 = noise.clone();
	meanFilter(res2, 7);
	imshow("7*7算术均值滤波", res2);

	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
}

void test4()
{
	Mat image, res1, res2, noise;
	image = imread("avatar.jpg", 0); // Read the file
	imshow("原始图像", image);

	noise = image.clone();
	addPepper(noise, 1000);
	Sleep(2000);
	addSalt(noise, 1000);
	imshow("添加1000个胡椒噪声+1000个盐噪声", noise);

	res1 = SelfAdaptMedianFilter(image);
	imshow("自适应中值滤波", res1);

	res2 = noise.clone();
	medeanFilter(res2, 7);
	imshow("7*7中值滤波", res2);

	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();
}

void test5()
{
	Mat image, res1, res2, noise;
	image = imread("avatar.jpg", 1); // Read the file
	imshow("原始图像", image);

	noise = addGaussianNoise(image);
	imshow("添加高斯噪声", noise);

	res1 = noise.clone();
	meanFilter(res1, 5);					//算术均值滤波器
	imshow("算术均值滤波器", res1);

	res2 = GeometryMeanFilter(noise);
	imshow("几何均值滤波器", res2);

	waitKey(0); // Wait for a keystroke in the window
	destroyAllWindows();

}
int main()
{
	test1();
	test2();
	test3();
	test4();
	test5();
	destroyAllWindows();
	return 0;
}