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
      cout << "Processing img " << line << endl;
      cv::Mat img;
      cv::imread(line, 0).convertTo(img, CV_32FC1);
      memcpy(imgData + counter * *w * *h, (float *)img.data, *w * *h * sizeof(float));
      img.release();
    }

    list.close();
    return imgData;
  }

  return NULL;
}