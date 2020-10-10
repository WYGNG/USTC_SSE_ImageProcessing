
#ifndef _MAIN_
#define _MAIN_
//
#include<iostream>
#include<string>
//
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
//
#include "src/6.cpp"


//
using namespace std;
using namespace cv;

#endif

int main(){
    VideoCapture capture(0);
    Mat frame;
    while(capture.isOpened()) {
        capture >> frame;
        if (frame.data)
        {
            Mat image = frame.clone();
            cv::cvtColor(image, image, COLOR_RGB2GRAY);

            for(int i = 0; i < image.rows; i++){
                for(int j = 0; j < image.cols; j++){

                    image.at<int8_t>(i, j) = 255 - image.at<int8_t>(i, j);

                }
            }
            imshow("VideoCapture", image);






        }

        waitKey(2);
    }

    return 0;
}