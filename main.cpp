#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"

#include "cuBoF.h"
#include "utils.h"

// TODO
// - Examine other kmeans algorithms
// - Remove duplicate SIFT keypoints?

int main(int argc, char **argv) {

  int numFeatures = 500;

  int w, h, numTrainingImages;

  float *imgData = loadTrainingDataFromList("train.txt", &numTrainingImages, &w, &h);

  cuBoF bag = cuBoF(numFeatures, numTrainingImages);
  bag.train(imgData, w, h);
  bag.save("tmp.bof");

  cv::Mat test = cv::Mat(480, 640, CV_32FC1, imgData);

  imwrite("test.jpg", test);

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

  cv::Mat img;
  cv::imread(argv[1], 0).convertTo(img, CV_32FC1);

  SiftData siftData;

  InitCuda(0);
  CudaImage cudaImg;
  cudaImg.Allocate(w, h, iAlignUp(w, bag.numDimensions), false, NULL, (float *)img.data);
  cudaImg.Download();

  float initBlur = 0.0f;
  float thresh = 5.0f;
  InitSiftData(siftData, 4096, true, true);
  ExtractSift(siftData, cudaImg, 5, initBlur, thresh, 0.0f);

  float *histogram = bag.vectorize(&siftData);

  cout << "Vectorized image histogram:" << endl;
  for (int i = 0; i < bag.numFeatures; i++) {
    cout << histogram[i] << " ";
  }
  cout << endl;

  free(histogram);
  FreeSiftData(siftData);
  */

  // cout << data << endl;
/*
  char *limgPath = argv[1];
  char *rimgPath = argv[2];

  cv::Mat limg, rimg;
  cv::imread(limgPath, 0).convertTo(limg, CV_32FC1);
  cv::imread(rimgPath, 0).convertTo(rimg, CV_32FC1);

  unsigned int w = limg.cols;
  unsigned int h = limg.rows;

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
  float *vldata = new float[128 * numData];

  for (int i = 0; i < numData; i++) {
    memcpy(vldata + 128 * i, lSiftData.h_data[i].data, 128 * sizeof(float));

    // fprintf(stderr, "Data %d: ", i);
    // for (int j = 0; j < 128; j++) {
    //   fprintf(stderr, "%0.4f ", data[i * 128 + j]);
    // }
    // fprintf(stderr, "\n");
  }

  cout << "Clustering SIFT..." << endl;

	VlKMeans *kMeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
  vl_kmeans_set_algorithm(kMeans, VlKMeansANN);
  vl_kmeans_init_centers_with_rand_data (kMeans, vldata, dimension, numData, numCenters);
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
  float numerator = log(numTrainingImages + 1);
  for (int i = 0; i < numCenters; i++) {
    int numTrainingImagesWithTerm = 0;
    if (lHistogram[i] > 0) numTrainingImagesWithTerm++;
    if (rHistogram[i] > 0) numTrainingImagesWithTerm++;

    IDFWeights[i] = numerator / max(numTrainingImagesWithTerm, 1);
  }

  cout << "Weighting histograms..." << endl;
  float lNorm = 0;
  float rNorm = 0;
  for (int i = 0; i < numCenters; i++) {
    lHistogram[i] *= IDFWeights[i];
    rHistogram[i] *= IDFWeights[i];

    lNorm += lHistogram[i] * lHistogram[i];
    rNorm += rHistogram[i] * rHistogram[i];
  }

  cout << "Normalizing histograms..." << endl;
  lNorm = sqrt(lNorm);
  rNorm = sqrt(rNorm);
  for (int i = 0; i < numCenters; i++) {
    lHistogram[i] /= lNorm;
    rHistogram[i] /= rNorm;
  }

  cout << "IDF weights: ";
  for (int i = 0; i < numCenters; i++) {
    cout << IDFWeights[i] << " ";
  }
  cout << endl;

  free(lHistogram);
  free(rHistogram);
  free(IDFWeights);
  free(vldata);
  vl_kmeans_delete(kMeans);
  vl_kdforest_delete(kdForest);
  FreeSiftData(lSiftData);
  FreeSiftData(rSiftData);
  limg.release();
  rimg.release();

  */

	return 0;
}