#pragma once
cv::Mat image_histogram(cv::Mat image, int* h);
void equalization_sequential(const char* image_path);
void equalization_parallel_omp(const char* image_path, bool Flag);
