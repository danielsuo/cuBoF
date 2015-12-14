#include <iostream>
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

int main(int argc, char **argv) {

	VL_PRINT("Hello world!\n");

  char *limgPath = argv[1];
  char *rimgPath = argv[2];

  cv::Mat limg, rimg;
  cv::imread(limgPath, 0).convertTo(limg, CV_32FC1);
  cv::imread(rimgPath, 0).convertTo(rimg, CV_32FC1);

  unsigned int w = limg.cols;
  unsigned int h = limg.rows;
  cout << "Image size = (" << w << "," << h << ")" << endl;

  std::cout << "Initializing data..." << std::endl;

  InitCuda(0);
  CudaImage img1, img2;
  img1.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)limg.data);
  img2.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float*)rimg.data);
  img1.Download();
  img2.Download();

  // Extract Sift features from images
  std::cout << "Extracting SIFT..." << std::endl;

  SiftData siftData1, siftData2;
  float initBlur = 0.0f;
  float thresh = 5.0f;
  InitSiftData(siftData1, 4096, true, true); 
  InitSiftData(siftData2, 4096, true, true);
  ExtractSift(siftData1, img1, 5, initBlur, thresh, 0.0f);
  ExtractSift(siftData2, img2, 5, initBlur, thresh, 0.0f);

  double energy;
  float *centers;
  vl_size numData = siftData1.numPts;
  vl_size numCenters = 20;
  vl_size dimension = 128;
  float *data = new float[128 * numData];

  for (int i = 0; i < numData; i++) {
    memcpy(data + 128 * i, siftData1.h_data[i].data, 128 * sizeof(float));

    // fprintf(stderr, "Data %d: ", i);
    // for (int j = 0; j < 128; j++) {
    //   fprintf(stderr, "%0.4f ", data[i * 128 + j]);
    // }
    // fprintf(stderr, "\n");
  }

  std::cout << "Clustering SIFT..." << std::endl;

	VlKMeans *kMeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
  vl_kmeans_set_algorithm(kMeans, VlKMeansElkan);
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

  std::cout << "Building vocabulary tree..." << std::endl;

  vl_size numTrees = 3;
  VlKDForest *kdForest = vl_kdforest_new(VL_TYPE_FLOAT, dimension, numTrees, VlDistanceL2);
  vl_kdforest_build(kdForest, numData, centers);

  VlKDForestSearcher *kdForestSearcher = vl_kdforest_new_searcher(kdForest);

  vl_size numNeighbors = 1;
  VlKDForestNeighbor kdForestNeighbor;

  float *query = centers + 10 * 128;
  vl_kdforest_query(kdForest, &kdForestNeighbor, numNeighbors, query);

  cout << "Neighbor: " << kdForestNeighbor.distance << " " << kdForestNeighbor.index << endl;

  free(data);
  vl_kmeans_delete(kMeans);
  vl_kdforest_delete(kdForest);
  // Don't need to free because kd_forest should delete searcher
  // vl_kdforestsearcher_delete(kdForestSearcher);
  FreeSiftData(siftData1);
  FreeSiftData(siftData2);
  limg.release();
  rimg.release();

	return 0;
}