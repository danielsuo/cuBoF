#include <vector>
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"

#include "cuBoF.h"
#include "utils.h"

// TODO
// - Examine other kmeans algorithms
// - Remove duplicate SIFT keypoints?

int main(int argc, char **argv) {
  cuBoF bag = cuBoF("tmp.bof");
  vector<float *>histograms;

  int w, h, numTrainingImages;

  float *imgData = loadTrainingDataFromList("test.txt", &numTrainingImages, &w, &h);

  for (int i = 0; i < numTrainingImages; i++) {
    cout << "Processing image " << i + 1 << " of " << numTrainingImages << endl;
    SiftData *siftData = extractFeaturesFromImage(imgData + i * w * h, w, h);
    float *histogram = bag.vectorize(siftData);

    float maxScore = 0;
    int maxIndex = -1;
    for (int j = 0; j < histograms.size(); j++) {
      float score = dot(histogram, histograms[j], bag.numFeatures);
      if (score > maxScore) {
        maxScore = score;
        maxIndex = j;
      }
    }

    cout << "Max score (" << maxScore << ") achieved at index " << maxIndex + 1 << endl;

    histograms.push_back(histogram);

    free(histogram);
    FreeSiftData(*siftData);

    cout << endl;
  }


  /*--- Train the bag of features -------------------------------------------- */

  // int numFeatures = 500;

  // int w, h, numTrainingImages;

  // float *imgData = loadTrainingDataFromList("train.txt", &numTrainingImages, &w, &h);

  // cuBoF bag = cuBoF(numFeatures, numTrainingImages);
  // bag.train(imgData, w, h);
  // bag.save("tmp.bof");

  // free(imgData);





  /*
  float *imgData = new float[numTrainingImages * w * h];

  for (int i = 0; i < numTrainingImages; i++) {
    char *path = argv[i+1];

    cv::Mat img;
    cv::imread(path, 0).convertTo(img, CV_32FC1);
    
    memcpy(imgData + i * w * h, (float *)img.data, w * h * sizeof(float));
    img.release();
  }

  cuBoF bag = cuBoF(numFeatures, numTrainingImages);
  bag.train(imgData, w, h);

  free(imgData);

  bag.save("tmp.bof");

  cuBoF bag2 = cuBoF("tmp.bof");

  for (int i = 0; i < bag.numFeatures; i++) {
    for (int j = 0; j < bag.numDimensions; j++) {
      // cout << i << " " << j << endl;
      // float test = bag.features[i * bag.numDimensions + j];
      if (bag.features[i * bag.numDimensions + j] != bag2.features[i * bag.numDimensions + j]) {
        cout << "AOOIDSJFOSIDJF " << endl;
      }
    }
    if (bag.weights[i] != bag2.weights[i]) {
      cout << "asdofijasodifjaosdijf" << endl;
    }
  }
  

  cuBoF bag = cuBoF("tmp.bof");

 
  */

	return 0;
}