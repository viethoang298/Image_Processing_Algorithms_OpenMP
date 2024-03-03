#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
#include <omp.h>
#include <chrono>
#include "image_gauss_filter.h"
using namespace std;
using namespace cv;

void average_blur(const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	//Mat result
	Mat blur(image.rows, image.cols, CV_8UC1);
	Mat difference(image.rows, image.cols, CV_8UC1);
	int innerMatrixIndex = 3;
	float sum, avarage;

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	//average blur

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			avarage = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += image.at<uchar>(i + a, j + b);
					}
				}
			}

			avarage = float(sum / float(innerMatrixIndex * innerMatrixIndex));
			///// asignacion
			blur.at<uchar>(i, j) = avarage;
		}
	}
	//Diffent image
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			difference.at<uchar>(i, j) = abs(image.at<uchar>(i, j) - blur.at<uchar>(i, j));
		}
	}
	int limite = 40;
	//Limit different image
	for (int i = 0; i < difference.rows; i++)
	{
		for (int j = 0; j < difference.cols; j++)
		{
			if (difference.at<uchar>(i, j) <= limite)
			{
				difference.at<uchar>(i, j) = 0;
			}
			else {
				difference.at<uchar>(i, j) = 255;
			}
		}
		cout << endl;
	}

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	namedWindow("blur", WINDOW_AUTOSIZE);
	imshow("blur", blur);
	namedWindow("diferencia", WINDOW_AUTOSIZE);
	imshow("diferencia", difference);
	waitKey(0);
}
void average_blur_omp(const char* image_path, bool Flag) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	//Mat result
	Mat blur(image.rows, image.cols, CV_8UC1);
	Mat difference(image.rows, image.cols, CV_8UC1);
	int innerMatrixIndex = 3;
	float sum, average;
	int get_num_threads, get_num_procs;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	
	//average blur
	#pragma omp parallel for private(sum, average)
	for (int i = 0; i < image.rows; i++)
	{
		get_num_threads = omp_get_num_threads();
		get_num_procs = omp_get_num_procs();
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			average = 0;

			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += image.at<uchar>(i + a, j + b);
					}
				}
			}

			average = float(sum / float(innerMatrixIndex * innerMatrixIndex));
			///// asignacion
			blur.at<uchar>(i, j) = average;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time omp: " << elapsed_seconds.count() << "s\n";
	std::cout << "Num threads: " << get_num_threads << endl;
	//Diffent image
#pragma omp parallel for
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			difference.at<uchar>(i, j) = abs(image.at<uchar>(i, j) - blur.at<uchar>(i, j));
		}
	}
	
	int limite = 40;
	//Limit different image
#pragma omp parallel for
	for (int i = 0; i < difference.rows; i++)
	{
		for (int j = 0; j < difference.cols; j++)
		{
			if (difference.at<uchar>(i, j) <= limite)
			{
				difference.at<uchar>(i, j) = 0;
			}
			else {
				difference.at<uchar>(i, j) = 255;
			}
		}
		cout << endl;
	}

	if (Flag) {
		namedWindow("blur_omp", WINDOW_AUTOSIZE);
		imshow("blur_omp", blur);
		namedWindow("diferencia_omp", WINDOW_AUTOSIZE);
		imshow("diferencia_omp", difference);
	}
	waitKey(0);
}
void median_filter(const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat filterMedian(image.rows, image.cols, CV_8UC1);
	int innerMatrixIndex = 3;
	float mediana = 0;
	vector<float> myVector;
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (int i = 0; i < image.rows; i++)
	{
		
		for (int j = 0; j < image.cols; j++)
		{
			mediana = 0;
			myVector.clear();
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					//agregarlos al vector
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						myVector.push_back(image.at<uchar>(i + a, j + b));
					}
				}
			}
			sort(myVector.begin(), myVector.end());
			mediana = myVector.at(myVector.size() / 2);
			///// asignacion
			filterMedian.at<uchar>(i, j) = mediana;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	
	namedWindow("filtroMediana", WINDOW_AUTOSIZE);
	imshow("filtroMediana", filterMedian);
	waitKey(0);
}

void median_filter_omp(const char* image_path, bool Flag) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat filterMedian(image.rows, image.cols, CV_8UC1);
	int innerMatrixIndex = 3;
	float mediana = 0;
	vector<float> myVector;
	int get_num_threads, get_num_procs;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
#pragma omp parallel for private (mediana, myVector)
	for (int i = 0; i < image.rows; i++)
	{
		get_num_threads = omp_get_num_threads();
		get_num_procs = omp_get_num_procs();
		for (int j = 0; j < image.cols; j++)
		{
			mediana = 0;
			myVector.clear();
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					//agregarlos al vector
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						myVector.push_back(image.at<uchar>(i + a, j + b));
					}
				}
			}
			sort(myVector.begin(), myVector.end());
			mediana = myVector.at(myVector.size() / 2);
			///// asignacion
			filterMedian.at<uchar>(i, j) = mediana;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time omp: " << elapsed_seconds.count() << "s\n";
	std::cout << "Num threads: " << get_num_threads << endl;
	if (Flag) {
		namedWindow("filtroMediana_omp", WINDOW_AUTOSIZE);
		imshow("filtroMediana_omp", filterMedian);
	}
	waitKey(0);
}