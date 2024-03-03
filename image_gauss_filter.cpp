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

void gauss_blur_3x3(const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat gaussiano_3x3(image.rows, image.cols, CV_8UC1);
	int innerMatrixIndex = 3;
	int gaussMatrix[3][3] = { { 1,2,1 },{ 2,4,2 },{ 1,2,1 } };
	int sum = 0;
	int avarage = 0;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			avarage = 0;
			int y = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				int x = 0;
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += int(image.at<uchar>(i + a, j + b) * gaussMatrix[y][x]);
					}
					x++;
				}
				y++;
			}
			/////promedio
			avarage = int(sum / 16);
			///// asignacion
			gaussiano_3x3.at<uchar>(i, j) = avarage;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	namedWindow("gaussiano_3x3", WINDOW_AUTOSIZE);
	imshow("gaussiano_3x3", gaussiano_3x3);
	waitKey(0);
}

void gauss_blur_3x3_omp(const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat gaussiano_3x3(image.rows, image.cols, CV_8UC1);
	int innerMatrixIndex = 3;
	int gaussMatrix[3][3] = { { 1,2,1 },{ 2,4,2 },{ 1,2,1 } };
	int sum = 0;
	int avarage = 0;
	int get_num_threads, get_num_procs;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	#pragma omp parallel for  private(sum,avarage)
	for (int i = 0; i < image.rows; i++)
	{
		get_num_threads = omp_get_num_threads();
		get_num_procs = omp_get_num_procs();
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			avarage = 0;
			int y = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				int x = 0;
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += int(image.at<uchar>(i + a, j + b) * gaussMatrix[y][x]);
					}
					x++;
				}
				y++;
			}
			/////promedio
			avarage = int(sum / 16);
			///// asignacion
			gaussiano_3x3.at<uchar>(i, j) = avarage;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time omp: " << elapsed_seconds.count() << "s\n";
	std::cout << "Number threads:" << get_num_threads << endl;
	std::cout << "Number procs:" << get_num_procs << endl;
	//namedWindow("gaussiano_3x3_omp", WINDOW_AUTOSIZE);
	//imshow("gaussiano_3x3_omp", gaussiano_3x3);
	waitKey(0);
}
void gauss_blur_5x5(const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat gaussiano_5x5(image.rows, image.cols, CV_8UC1);
	int innerMatrixIndex = 5;
	int gaussMatrix[5][5] = { { 1,4,7,4,1 },{ 4,16,26,16,4 },{ 7,26,41,26,7 },{ 4,16,26,16,4 },{ 1,4,7,4,1 } };
	int sum = 0;
	int avarage = 0;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			avarage = 0;
			int y = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				int x = 0;
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += int(image.at<uchar>(i + a, j + b) * gaussMatrix[y][x]);
					}
					x++;
				}
				y++;
			}
			/////promedio
			avarage = int(sum / 273);
			///// asignacion
			gaussiano_5x5.at<uchar>(i, j) = avarage;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	namedWindow("gaussiano_5x5", WINDOW_AUTOSIZE);
	imshow("gaussiano_5x5", gaussiano_5x5);
	waitKey(0);

}
void gauss_blur_5x5_omp(const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat gaussiano_5x5(image.rows, image.cols, CV_8UC1);
	int innerMatrixIndex = 5;
	int gaussMatrix[5][5] = { { 1,4,7,4,1 },{ 4,16,26,16,4 },{ 7,26,41,26,7 },{ 4,16,26,16,4 },{ 1,4,7,4,1 } };
	int sum = 0;
	int avarage = 0;
	int get_num_threads, get_num_procs;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	#pragma omp parallel for private(sum, avarage)
	for (int i = 0; i < image.rows; i++)
	{
		get_num_threads = omp_get_num_threads();
		get_num_procs = omp_get_num_procs();
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			avarage = 0;
			int y = 0;
			for (int a = -(innerMatrixIndex / 2); a <= innerMatrixIndex / 2; a++)
			{
				int x = 0;
				for (int b = -(innerMatrixIndex / 2); b <= innerMatrixIndex / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += int(image.at<uchar>(i + a, j + b) * gaussMatrix[y][x]);
					}
					x++;
				}
				y++;
			}
			/////promedio
			avarage = int(sum / 273);
			///// asignacion
			gaussiano_5x5.at<uchar>(i, j) = avarage;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	std::cout << "Number threads:" << get_num_threads << endl;
	std::cout << "Number procs:" << get_num_procs << endl;
	//namedWindow("gaussiano_5x5", WINDOW_AUTOSIZE);
	//imshow("gaussiano_5x5", gaussiano_5x5);
	waitKey(0);
}
void gauss_blur_25x25(const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat gaussiano_25x25(image.rows, image.cols, CV_8UC1);
	float gauss[25][25] = { 0 };
	int x0 = 25 / 2;
	int y0 = 25 / 2;
	int sigma = 3;
	float pi = 3.1416;
	float totalFilter = 0;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	for (int i = 0; i < 25; i++) {
		for (int j = 0; j < 25; j++) {
			int cX = i - x0;
			int cY = y0 - j;
			float up = (cX * cX) + (cY * cY);
			float down = 2 * (sigma * sigma);
			float exp1 = exp(-(up) / (down));
			float constant = 1.0 / (sigma * sigma * 2 * pi);
			gauss[i][j] = constant * exp1;
			totalFilter += constant * exp1;
		}
	}

	float sum = 0;
	int avarage = 0;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			avarage = 0;
			int y = 0;
			for (int a = -(25 / 2); a <= 25 / 2; a++)
			{
				int x = 0;
				for (int b = -(25 / 2); b <= 25 / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += int(image.at<uchar>(i + a, j + b)) * gauss[y][x];
					}
					x++;
				}
				y++;
			}
			/////promedio
			avarage = int(sum / totalFilter);
			///// asignacion
			gaussiano_25x25.at<uchar>(i, j) = avarage;
		}
	}
	
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	namedWindow("gaussiano_25x25", WINDOW_AUTOSIZE);
	imshow("gaussiano_25x25", gaussiano_25x25);
	waitKey(0);
}


void gauss_blur_25x25_omp(const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat gaussiano_25x25(image.rows, image.cols, CV_8UC1);
	float gauss[25][25] = { 0 };
	int x0 = 25 / 2;
	int y0 = 25 / 2;
	int sigma = 3;
	float pi = 3.1416;
	float totalFilter = 0;
	int get_num_threads, get_num_procs;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	#pragma omp parallel for reduction(+:totalFilter)   //totalFilter is shared variable 
	for (int i = 0; i < 25; i++) {
		get_num_threads = omp_get_num_threads();
		get_num_procs   = omp_get_num_procs();
		for (int j = 0; j < 25; j++) {
			int cX = i - x0;
			int cY = y0 - j;
			float up = (cX * cX) + (cY * cY);
			float down = 2 * (sigma * sigma);
			float exp1 = exp(-(up) / (down));
			float constant = 1.0 / (sigma * sigma * 2 * pi);
			gauss[i][j] = constant * exp1;
			totalFilter += constant * exp1;
		}
	}

	float sum = 0;
	int avarage = 0;
	#pragma omp parallel for private(sum, avarage)
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			avarage = 0;
			int y = 0;
			for (int a = -(25 / 2); a <= 25 / 2; a++)
			{
				int x = 0;
				for (int b = -(25 / 2); b <= 25 / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += int(image.at<uchar>(i + a, j + b)) * gauss[y][x];
					}
					x++;
				}
				y++;
			}
			/////promedio
			avarage = int(sum / totalFilter);
			///// asignacion
			gaussiano_25x25.at<uchar>(i, j) = avarage;
		}
	}

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time omp: " << elapsed_seconds.count() << "s\n";
	std::cout << "Number threads:" << get_num_threads << endl;
	std::cout << "Number procs:" << get_num_procs << endl;
	//namedWindow("gaussiano_25x25_omp", WINDOW_AUTOSIZE);
	//imshow("gaussiano_25x25_omp", gaussiano_25x25);
	waitKey(0);
}


void gauss_blur_nxn_omp(int n, const char* image_path) {
	Mat image = imread(image_path, IMREAD_GRAYSCALE);
	Mat gaussiano(image.rows, image.cols, CV_8UC1);
	float** gauss = new float*[n];
	for (int i = 0; i < n; i++)
		gauss[i] = new float[n];
	int x0 = n / 2;
	int y0 = n / 2;
	int sigma = 3;
	float pi = 3.1416;
	float totalFilter = 0;
	int get_num_threads, get_num_procs;
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	#pragma omp parallel for reduction(+:totalFilter)
	for (int i = 0; i < n; i++) {
		get_num_threads = omp_get_num_threads();
		get_num_procs = omp_get_num_procs();

		for (int j = 0; j < n; j++) {
			int cX = i - x0;
			int cY = y0 - j;
			float up = (cX * cX) + (cY * cY);
			float down = 2 * (sigma * sigma);
			float exp1 = exp(-(up) / (down));
			float constant = 1.0 / (sigma * sigma * 2 * pi);
			gauss[i][j] = constant * exp1;
			totalFilter += constant * exp1;
		}
	}

	float sum = 0;
	int avarage = 0;
	#pragma omp parallel for private (sum, avarage)
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			sum = 0;
			avarage = 0;
			int y = 0;
			for (int a = -(n / 2); a <= n / 2; a++)
			{
				int x = 0;
				for (int b = -(n / 2); b <= n / 2; b++)
				{
					////// suma
					if ((i + a) >= 0 &&
						(i + a) < image.rows &&
						(j + b) >= 0 &&
						(j + b) < image.cols)
					{
						sum += int(image.at<uchar>(i + a, j + b)) * gauss[y][x];
					}
					x++;
				}
				y++;
			}
			/////promedio
			avarage = int(sum / totalFilter);
			///// asignacion
			gaussiano.at<uchar>(i, j) = avarage;
		}
	}
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
	std::cout << "Number threads:" << get_num_threads << endl;
	std::cout << "Number procs:" << get_num_procs << endl;
	namedWindow("original image", WINDOW_AUTOSIZE);
	imshow("original image", image);
	namedWindow("gaussiano", WINDOW_AUTOSIZE);
	imshow("gaussiano", gaussiano);
	waitKey(0);
	for (int i = 0; i < n; i++)
		delete[] gauss[i];
	delete[] gauss;
}