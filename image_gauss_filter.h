#pragma once
void gauss_blur_3x3(const char* image_path);
void gauss_blur_5x5(const char* image_path);
void gauss_blur_25x25(const char* image_path);
void gauss_blur_3x3_omp(const char* image_path);
void gauss_blur_5x5_omp(const char* image_path);
void gauss_blur_25x25_omp(const char* image_path);
void gauss_blur_nxn_omp(int n, const char* image_path);