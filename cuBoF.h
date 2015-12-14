#ifndef CUBOF_H
#define CUBOF_H

#include <vector>

extern "C" {
  #include <vl/generic.h>
}

using namespace std;

class cuBoF {
public:
  int num_images;

  cuBoF();
  ~cuBoF();

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

  // Quantize vector of features into visual word vocabulary
  void computeHistogram();

};

#endif