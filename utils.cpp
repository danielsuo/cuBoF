#include "utils.h"

float *loadTrainingDataFromList(const char *trainingDataList, int *numTrainingImages, int *w, int *h) {

  string line;
  ifstream list(trainingDataList);

  if (list.is_open()) {
    list >> *numTrainingImages;
    list >> *w;
    list >> *h;

    cout << "Found " << *numTrainingImages << " images with width " << *w << " and height " << *h << endl;  
    cout << "Need to allocate " << *w * *h * *numTrainingImages << " bytes of memory" << endl;

    float *imgData = (float *)malloc(*w * *h * *numTrainingImages * sizeof(float));

    int counter = 0;

    // Finish off empty line
    getline(list, line);

    while (getline(list, line)) {
      cv::Mat img;
      cv::imread(line, 0).convertTo(img, CV_32FC1);

      memcpy(imgData + counter * *w * *h, (float *)img.data, *w * *h * sizeof(float));

      counter++;
      img.release();
    }

    list.close();
    return imgData;
  }

  return NULL;
}

SiftData *extractFeaturesFromImage(float *imgData, int w, int h) {
  SiftData *siftData = new SiftData();

  InitCuda(0);
  CudaImage cudaImg;
  cudaImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, imgData);
  cudaImg.Download();

  float initBlur = 0.0f;
  float thresh = 5.0f;
  InitSiftData(*siftData, 4096, true, true);
  ExtractSift(*siftData, cudaImg, 5, initBlur, thresh, 0.0f);

  cout << "Extracted " << siftData->numPts << " key points" << endl;

  return siftData;
}

float dot(float *hist1, float *hist2, int numBins) {
  float sum = 0;

  for (int i = 0; i < numBins; i++) {
    sum += hist1[i] * hist2[i];
  }

  return sum;
}

float intersect(float *hist1, float *hist2, int numBins) {
  float sum = 0;
  for (int i = 0; i < numBins; i++) {
    sum += min(hist1[i], hist2[i]);
  }

  return sum;
}