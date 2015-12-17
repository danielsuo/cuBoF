#ifndef CUBOFUTILS_H
#define CUBOFUTILS_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace std;

float *loadTrainingDataFromList(const char *trainingDataList, int *numTrainingImages, int *w, int *h);

#endif