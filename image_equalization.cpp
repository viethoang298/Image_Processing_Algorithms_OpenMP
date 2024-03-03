// opencvtest.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>
#include <chrono>
#include "image_equalization.h"

using namespace std;
using namespace cv;

cv::Mat image_histogram(cv::Mat image, int* h) {
	#pragma omp parallel for collapse(2)  //nested loop
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
	        int index = (int)(image.at<uchar>(i, j));
			#pragma omp atomic            // racing condition when accessing memory
			h[index]++;
		}
	}
	int maxH = 0;
	#pragma omp parallel for 
	for (int i = 0; i < 255; i++) {
		if (h[i] > maxH) {
			maxH = h[i];
		}
	}
    Mat histogr(750, 750, CV_8U, Scalar(0));
    int inc = 750 / 256;
    #pragma omp parallel for 
	for (int i = 0; i < 255; i++) {
		rectangle(histogr, Point(inc * i, histogr.rows), Point((inc * (i + 1) - 1), histogr.rows - ((h[i] * histogr.rows) / (maxH))), Scalar(255, 255, 255, 0), cv::FILLED);
	}
	return histogr;
}

void equalization_sequential(const char* image_path) {
	int arr[256] = {0};
	float arr2[256] = { 0 };
	float arr3[256] = { 0 };
	Mat image = imread(image_path, IMREAD_GRAYSCALE);  // Ma tran (m*n) diem anh 
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	// Origin Histogram
	Mat histog(750, 750, CV_8U, Scalar(0));
    histog = image_histogram(image, arr);
	int inc = 750 / 256;

	
	//*********** Equalization Algorithm *************
	int height = image.rows;
	int width = image.cols;
	Mat myMat1(height, width, CV_8U, Scalar(0));  //Khoi tao ma tran 8 bit voi gia tri ban dau la 0

	//-- PMF - probability mass function - Ham Thong ke tan suat n/N
	float total = image.cols * image.rows;
	for (int i = 0; i < 255; i++)
	{
		arr2[i] = float(arr[i]) / total;
	}
	arr3[0] = arr2[0];

	//-- CDF - Cumulative distribution function - Phan phoi tich luy
	for (int i = 1; i < 255; i++)
	{
		
		arr3[i] = arr2[i] + arr3[i - 1];
	}

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			myMat1.at<uchar>(i, j) = floor((256 - 1) * arr3[image.at<uchar>(i, j)]);
		}
	}
	//************
	// histogram equalized image
	int h2[256] = { 0 };
	Mat histog2(750, 750, CV_8U, Scalar(0));
	histog2 = image_histogram(myMat1, h2);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	
	namedWindow("Histogram Original Image", WINDOW_AUTOSIZE);
	imshow("Histogram Original Image", histog);

	namedWindow("Histogram Equalized Image", WINDOW_AUTOSIZE);
	imshow("Histogram Equalized Image", histog2);

	namedWindow("Equalized Image", WINDOW_AUTOSIZE);
	imshow("Equalized Image", myMat1);
	waitKey(0);
}

void equalization_parallel_omp(const char* image_path, bool Flag) {
	int arr[256] = { 0 };
	float arr2[256] = { 0 };
	float arr3[256] = { 0 };
	int get_num_threads, get_num_procs;
	Mat image = imread(image_path, IMREAD_GRAYSCALE);  // Ma tran (m*n) diem anh 
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	// Origin Histogram
	Mat histog(750, 750, CV_8U, Scalar(0));
	histog = image_histogram(image, arr);
	int inc = 750 / 256;


	//*********** Equalization Algorithm *************
	int height = image.rows;
	int width = image.cols;
	Mat myMat1(height, width, CV_8U, Scalar(0));  ////Khoi tao ma tran 8 bit voi gia tri ban dau la 0

	//-- PMF - probability mass function - Ham Thong ke tan suat n/N   
	float total = image.cols * image.rows;
    #pragma omp parallel for 
	for (int i = 0; i < 255; i++){
		get_num_threads = omp_get_num_threads();
		get_num_procs = omp_get_num_procs();
		arr2[i] = float(arr[i]) / total;
	}
	arr3[0] = arr2[0];

	//-- CDF - Cumulative distribution function - Phan phoi tich luy
	//#pragma omp parallel for  
	for (int i = 1; i < 255; i++){
		//#pragma omp critical
		arr3[i] = arr2[i] + arr3[i - 1];
	}
	 
	#pragma omp parallel for
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			myMat1.at<uchar>(i, j) = floor((256 - 1) * arr3[image.at<uchar>(i, j)]);
		}
	}
	//************
	// histogram equalized image
	int h2[256] = { 0 };
	Mat histog2(750, 750, CV_8U, Scalar(0));
	histog2 = image_histogram(myMat1, h2);
	
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time omp: " << elapsed_seconds.count() << "s\n";
	std::cout << "Number threads:" << get_num_threads << endl;
	std::cout << "Number procs:" << get_num_procs << endl;

	if (Flag) {
		namedWindow("Histogram Original Image omp", WINDOW_AUTOSIZE);
		imshow("Histogram Original Image omp", histog);
		namedWindow("Histogram Equalized Image omp", WINDOW_AUTOSIZE);
		imshow("Histogram Equalized Image omp", histog2);
		namedWindow("Equalized Image omp", WINDOW_AUTOSIZE);
		imshow("Equalized Image omp", myMat1);
	}
	waitKey(0);
}

