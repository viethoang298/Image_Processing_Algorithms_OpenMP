#pragma once
void average_blur(const char* image_path);
void average_blur_omp(const char* image_path, bool Flag);
void median_filter(const char* image_path);
void median_filter_omp(const char* image_path, bool Flag);