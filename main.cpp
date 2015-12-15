#include <iostream>
#include <ctime>
#include <math.h>
#include <algorithm> 
#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"

#include "cuBoF.h"

extern "C" {
  #include <vl/generic.h>
  #include <vl/kmeans.h>
  #include <vl/sift.h>
}

#include "lib/cuSIFT/cudaSift.h"

using namespace std;

// TODO
// - Examine other kmeans algorithms
// - Remove duplicate SIFT keypoints?

int main(int argc, char **argv) {

	VL_PRINT("Hello world!\n");

  int numFrames = argc - 1;



  char *limgPath = argv[1];
  char *rimgPath = argv[2];

  cv::Mat limg, rimg;
  cv::imread(limgPath, 0).convertTo(limg, CV_32FC1);
  cv::imread(rimgPath, 0).convertTo(rimg, CV_32FC1);

  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  cout << "Image size = (" << w << "," << h << ")" << endl;

  cout << "Initializing data..." << endl;

  InitCuda(0);
  CudaImage lCudaImg, rCudaImg;
  lCudaImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  rCudaImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
  lCudaImg.Download();
  rCudaImg.Download();

  // Extract Sift features from images
  cout << "Extracting SIFT..." << endl;

  SiftData lSiftData, rSiftData;
  float initBlur = 0.0f;
  float thresh = 5.0f;
  InitSiftData(lSiftData, 4096, true, true); 
  InitSiftData(rSiftData, 4096, true, true);
  ExtractSift(lSiftData, lCudaImg, 5, initBlur, thresh, 0.0f);
  ExtractSift(rSiftData, rCudaImg, 5, initBlur, thresh, 0.0f);

  double energy;
  float *centers;
  vl_size numData = lSiftData.numPts;
  vl_size numCenters = 20;
  vl_size dimension = 128;
  float *data = new float[128 * numData];

  for (int i = 0; i < numData; i++) {
    memcpy(data + 128 * i, lSiftData.h_data[i].data, 128 * sizeof(float));

    // fprintf(stderr, "Data %d: ", i);
    // for (int j = 0; j < 128; j++) {
    //   fprintf(stderr, "%0.4f ", data[i * 128 + j]);
    // }
    // fprintf(stderr, "\n");
  }

  cout << "Clustering SIFT..." << endl;

	VlKMeans *kMeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
  vl_kmeans_set_algorithm(kMeans, VlKMeansANN);
  vl_kmeans_init_centers_with_rand_data (kMeans, data, dimension, numData, numCenters);
  vl_kmeans_set_max_num_iterations (kMeans, 100);
  energy = vl_kmeans_get_energy(kMeans);
  centers = (float *)vl_kmeans_get_centers(kMeans);

  // for (int i = 0; i < numCenters; i++) {
  //   fprintf(stderr, "Center %d: ", i);
  //   for (int j = 0; j < 128; j++) {
  //     fprintf(stderr, "%0.4f ", centers[i * 128 + j]);
  //   }
  //   fprintf(stderr, "\n");
  // }

  cout << "Building vocabulary tree..." << endl;

  vl_size numTrees = 3;
  VlKDForest *kdForest = vl_kdforest_new(VL_TYPE_FLOAT, dimension, numTrees, VlDistanceL2);
  vl_size maxNumComparisons = 500;
  vl_kdforest_set_max_num_comparisons(kdForest, maxNumComparisons);
  vl_kdforest_build(kdForest, numData, centers);

  // VlKDForestSearcher *kdForestSearcher = vl_kdforest_new_searcher(kdForest);

  cout << "Building histogram..." << endl;

  clock_t start = clock();

  int *lHistogram = new int[numCenters]();
  cout << "Num points (l): " << lSiftData.numPts << endl;
  for (int i = 0; i < lSiftData.numPts; i++) {
    vl_size numNeighbors = 1;
    VlKDForestNeighbor kdForestNeighbor;
    float *query = lSiftData.h_data[i].data;
    vl_kdforest_query(kdForest, &kdForestNeighbor, numNeighbors, query);
    lHistogram[kdForestNeighbor.index]++;
  }

  cout << "Histogram: ";
  int lsum = 0;
  for (int i = 0; i < numCenters; i++) {
    cout << lHistogram[i] << " ";
    lsum += lHistogram[i];
  }
  cout << endl;
  cout << "Total points: " << lsum << endl;

  int *rHistogram = new int[numCenters]();
  cout << "Num points(r): " << rSiftData.numPts << endl;
  for (int i = 0; i < rSiftData.numPts; i++) {
    vl_size numNeighbors = 1;
    VlKDForestNeighbor kdForestNeighbor;
    float *query = rSiftData.h_data[i].data;
    vl_kdforest_query(kdForest, &kdForestNeighbor, numNeighbors, query);
    rHistogram[kdForestNeighbor.index]++;
  }
  
  cout << "Histogram: ";
  int rsum = 0;
  for (int i = 0; i < numCenters; i++) {
    cout << rHistogram[i] << " ";
    rsum += rHistogram[i];
  }
  cout << endl;
  cout << "Total points: " << rsum << endl;

  double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
  cout << "Building histogram took " << duration * 1000 << "ms." << endl;

  cout << "Computing IDF weights..." << endl;
  float *IDFWeights = new float[numCenters];
  float numerator = log(numFrames + 1);
  for (int i = 0; i < numCenters; i++) {
    int numFramesWithTerm = 0;
    if (lHistogram[i] > 0) numFramesWithTerm++;
    if (rHistogram[i] > 0) numFramesWithTerm++;

    IDFWeights[i] = numerator / max(numFramesWithTerm, 1);
  }

  cout << "Weighting and normalizing histograms..." << endl;
  for (int i = 0; i < numCenters; i++) {
    // lHistogram[i] 
  }

  cout << "IDF weights: ";
  for (int i = 0; i < numCenters; i++) {
    cout << IDFWeights[i] << " ";
  }
  cout << endl;

  free(lHistogram);
  free(rHistogram);
  free(IDFWeights);
  free(data);
  vl_kmeans_delete(kMeans);
  vl_kdforest_delete(kdForest);
  FreeSiftData(lSiftData);
  FreeSiftData(rSiftData);
  limg.release();
  rimg.release();

	return 0;
}