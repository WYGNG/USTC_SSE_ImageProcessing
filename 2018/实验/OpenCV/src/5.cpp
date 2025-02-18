//
// Created by XQ on 2019-03-29.
//

//https://www.jianshu.com/p/1c9ddc9a7b38
//https://blog.csdn.net/qq_37059483/article/details/77979910
//https://blog.csdn.net/gxiaoyaya/article/details/72460483

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

bool DFTAndIDFT(string &src, int flag);
bool idealLowPassFilterOrHighPassFilter(string &src, int flag, double f,int model);
bool idealHighPassFilter(string &src, int flag, double f);
bool butterworthLowPassFilterOrHighPassFilter(string &src, int flag, double f, int n, int model);





/*int main(){
    string str = "/Volumes/数据/图片/2k/lostwall.jpg";
    cout << "1.DFTAndIDFT:" << DFTAndIDFT(str,0) << endl;
    double f;
    int n;

    while (cout << "输入截止频率f 与 n:" && cin >> f >> n){
        cout << "2.1 idealLowPassFilter:" << idealLowPassFilterOrHighPassFilter(str, 0, f, 0) << endl;
        cout << "2.2 idealHighPassFilter:" << idealLowPassFilterOrHighPassFilter(str, 0, f, 1) << endl;
        cout << "3.1 ButterworthLowPassFilter:" << butterworthLowPassFilterOrHighPassFilter(str, 0, f, n, 0) << endl;
        cout << "3.2 ButterworthHighPassFilter:" << butterworthLowPassFilterOrHighPassFilter(str, 0, f, n, 1) << endl;
    }

    return 0;
}*/





/*灰度图像的 DFT 和 IDFT。
 *
 *具体内容：利用 OpenCV 提供的 cvDFT 函数对图像进行 DFT 和 IDFT 变换.
 * 在图像处理领域，通过DFT可以将图像转换到频域，实现高通和低通滤波；
 * 还可以利用矩阵的卷积运算等同于其在频域的乘法运算从而优化算法降低运算量，即先将图像转换到频域，然后做完乘法运算后，
 * 再转换到图像域，opencv中的模板匹配就利用了这一特性降低运算量。
 *
 * */

bool DFTAndIDFT(string &src, int flag){
    Mat image = imread(src,flag);
    imshow("input", image);
    int w = getOptimalDFTSize(image.cols);
    int h = getOptimalDFTSize(image.rows);
    Mat padded;
    copyMakeBorder(image, padded, 0, h - image.rows, 0, w - image.cols, BORDER_CONSTANT, Scalar::all(0));//填充图像保存到padded中
    Mat plane[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };//创建通道
    Mat complexIm;
    merge(plane, 2, complexIm);//合并通道
    dft(complexIm, complexIm);//进行傅立叶变换，结果保存在自身
    split(complexIm, plane);//分离通道
    magnitude(plane[0], plane[1], plane[0]);//获取幅度图像，0通道为实数通道，1为虚数，因为二维傅立叶变换结果是复数
    int cx = padded.cols / 2;
    int cy = padded.rows / 2;//以下的操作是移动图像，左上与右下交换位置，右上与左下交换位置
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
    /**/
    Mat _complexim;
    complexIm.copyTo(_complexim);//把变换结果复制一份，进行逆变换，也就是恢复原图
    Mat iDft[] = { Mat::zeros(plane[0].size(),CV_32F),Mat::zeros(plane[0].size(),CV_32F) };//创建两个通道，类型为float，大小为填充后的尺寸
    idft(_complexim, _complexim);//傅立叶逆变换
    split(_complexim, iDft);//结果貌似也是复数
    magnitude(iDft[0], iDft[1], iDft[0]);//分离通道，主要获取0通道
    normalize(iDft[0], iDft[0], 1, 0, CV_MINMAX);//归一化处理，float类型的显示范围为0-1,大于1为白色，小于0为黑色
    imshow("IDFT", iDft[0]);//显示逆变换
    /**/
    plane[0] += Scalar::all(1);//傅立叶变换后的图片不好分析，进行对数处理，结果比较好看
    log(plane[0], plane[0]);
    normalize(plane[0], plane[0], 1, 0, CV_MINMAX);
    imshow("DFT", plane[0]);
    waitKey(0);
    destroyAllWindows();


    return true;
}

/*利用理想高通和低通滤波器对灰度图像进行频域滤波
 *
 * 具体内容：利用 cvDFT 函数实现 DFT，在频域上利用理想高通和低通滤波器进行滤波，并把滤波过后的图像显示在屏幕上（观察振铃现象），要求截止频率可输入。
 *
 * */
bool idealLowPassFilterOrHighPassFilter(string &src, int flag, double f, int model){
    Mat image = imread(src, 0); // Read the file
    imshow("input", image);
    Mat img = image.clone();
    //cvtColor(src, img, CV_BGR2GRAY);
    //调整图像加速傅里叶变换
    if(model == 0) cout << "低通 ";
    if(model == 1) cout << "高通 ";
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
        auto * data = mag.ptr<double>(y);
        for (int x = 0; x < mag.cols; x++) {
            double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));

            if(model == 0){
                if(d>f) data[x] = 0;//低通
            } else if(model == 1){
                if(d<f) data[x] = 0;//高通
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

    imshow("idealLowPassFilterOrHighPassFilter", invDFTcvt);
    waitKey(0);
    destroyAllWindows();


    return true;
}

/*利用布特沃斯高通和低通滤波器对灰度图像进行频域滤波。
 *
 * 具体内容：利用 cvDFT 函数实现 DFT，在频域上进行利用布特沃斯高通和低通滤波器进行滤波，并把滤波过后的图像显示在屏幕上（观察振铃现象），要求截止频率和 n 可输入。
 *
 * */
bool butterworthLowPassFilterOrHighPassFilter(string &src, int flag, double f, int n, int model){
    Mat image = imread(src, 0); // Read the file
    imshow("原始图像", image);

    //H = 1 / (1+(D/D0)^2n)
    Mat img = image.clone();
    if(model == 0) cout << "低通 ";
    if(model == 1) cout << "高通 ";
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
        auto * data = mag.ptr<double>(y);
        for (int x = 0; x < mag.cols; x++)
        {
            //cout << data[x] << endl;
            double d = sqrt(pow((y - cy), 2) + pow((x - cx), 2));
            //cout << d << endl;
            double h = 0;

            if(model == 0)  h = 1.0 / (1 + pow(d / f, 2 * n));
            if(model == 1)  h = 1.0 / (1 + pow(f / d, 2 * n));
            if (h <= 0.5)
            {
                data[x] = 0;
            }

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
    imshow("butterworthLowPassFilterOrHighPassFilter", invDFTcvt);

    waitKey(0);
    destroyAllWindows();
    return true;
}


