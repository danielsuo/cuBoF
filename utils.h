#ifndef CUBOFUTILS_H
#define CUBOFUTILS_H

#include <random>
#include <algorithm>
#include <iterator>
#include <functional>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"

#include "cuSIFT/cudaSift.h"

using namespace std;
using namespace cv;

float *loadTrainingDataFromList(const char *trainingDataList, int *numTrainingImages, int *w, int *h);
SiftData *extractFeaturesFromImage(float *imgData, int w, int h);
float dot(float *hist1, float *hist2, int numBins);
float intersect(float *hist1, float *hist2, int numBins);
vector<int> getRandomIntVector(int lower, int upper, int length);
float *getRowSubset(int total, int subset, int dim, float *data);
Mat combineMatchedImages(Mat img1, Mat img2);

#endif