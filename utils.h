#ifndef CUBOFUTILS_H
#define CUBOFUTILS_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"

#include "lib/cuSIFT/cudaSift.h"

using namespace std;
using namespace cv;

float *loadTrainingDataFromList(const char *trainingDataList, int *numTrainingImages, int *w, int *h);
SiftData *extractFeaturesFromImage(float *imgData, int w, int h);
float dot(float *hist1, float *hist2, int numBins);
float intersect(float *hist1, float *hist2, int numBins);

#endif