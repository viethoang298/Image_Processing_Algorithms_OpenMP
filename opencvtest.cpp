//headers, librerias
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>
#include <omp.h>
#include "image_avarage_median_filter.h"
#include "image_gauss_filter.h"
#include "image_equalization.h"
using namespace std;
using namespace cv;


/*

int main() {
    std::string input_folder_path = "E:/TVHOANG/Courses/opencvtest/image/input/360x360/*.png"; 
    std::string output_folder_path = "E:/TVHOANG/Courses/opencvtest/image/output/360x360/myim.png";

	
	vector<cv::String> fn;
	vector<cv::Mat> data;
	cv::glob(input_folder_path, fn, true); // recurse
	for (size_t k = 0; k < fn.size(); ++k)
	{
		cv::Mat im = cv::imread(fn[k], IMREAD_GRAYSCALE);
		if (im.empty()) continue; //only proceed if sucsessful
		// you probably want to do some preprocessing
		data.push_back(im);
	}
	
	bool check = imwrite(output_folder_path, data[0]);

	
	namedWindow("Equalized Image omp", WINDOW_AUTOSIZE);
	imshow("Equalized Image omp", data[0]);
    return 0;
}

*/
int main()
{	const char* image_path_1 = "E:/TVHOANG/Courses/opencvtest/image/input/360x360/360x360.png";
	Mat image_1 = imread(image_path_1, IMREAD_GRAYSCALE);
	const char* image_path_2 = "E:/TVHOANG/Courses/opencvtest/image/input/640x640/640x640.jpg";
	Mat image_2 = imread(image_path_2, IMREAD_GRAYSCALE);
	const char* image_path_3 = "E:/TVHOANG/Courses/opencvtest/image/input/720x720/720x720.png";
	Mat image_3 = imread(image_path_3, IMREAD_GRAYSCALE);
	omp_set_num_threads(6);
	cout << "1" << endl; average_blur_omp(image_path_3, false);
	cout << "2" << endl; median_filter_omp(image_path_3, false);
	cout << "3" << endl; equalization_parallel_omp(image_path_3, false);
	cout << "4" << endl; gauss_blur_3x3_omp(image_path_3);
	cout << "5" << endl; gauss_blur_5x5_omp(image_path_3);
	cout << "6" << endl; gauss_blur_25x25_omp(image_path_3);
	return 0;
	}



