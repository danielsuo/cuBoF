#ifndef CUBOF_H
#define CUBOF_H

#include <iostream>
#include <ctime>
#include <math.h>
#include <algorithm> 
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"

extern "C" {
  #include <vl/kmeans.h>
}

#include "lib/cuSIFT/cudaSift.h"

using namespace std;

class cuBoF {
public:
  int numTrainingImages;
  int numFeatures;

  // These are k-means for clustered SIFT points from training phase
  float *features;

  // Hold inverse document frequency weights
  float *weights;

  // Hold the kmeans and the kdforest
  VlKMeans *kMeans;
  VlKDForest *kdForest;
  
  cuBoF(int numFeatures, int numTrainingImages);
  ~cuBoF();

  void train(char **paths);

  // Quantize vector of features into visual word vocabulary
  void quantize(int numSiftPoints, float *siftPointHistograms);


  void loadImages();

  /* Training -------------------------------------------------------------- */

  // Extra features on given image
  void extractFeaturesFromImage();

  // Extract all features and store
  void extractFeatures();

  // Create visual word vocabulary
  void clusterFeatures();


  /* Testing --------------------------------------------------------------- */

  void loadModel();

  
};

#endif