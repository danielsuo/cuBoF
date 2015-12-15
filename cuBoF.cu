#include "cuBoF.h"

cuBoF::cuBoF(int numFeatures, int numTrainingImages) {
  this->numFeatures = numFeatures;
  this->numTrainingImages = numTrainingImages;

  numDimensions = 128;

  features = new float[numDimensions * numFeatures];
  weights = new float[numFeatures];
}

void cuBoF::train(char **paths) {
  SiftData *siftData = new SiftData[numTrainingImages];
  unsigned int totalNumSIFT = 0;

  for (int i = 0; i < numTrainingImages; i++) {
    cout << "Processing image " << i << " of " << numTrainingImages << endl;

    char *path = paths[i];
    cv::Mat img;
    cv::imread(path, 0).convertTo(img, CV_32FC1);
    unsigned int w = img.cols;
    unsigned int h = img.rows;
    cout << "Image size = (" << w << "," << h << ")" << endl;

    InitCuda(0);
    CudaImage cudaImg;
    cudaImg.Allocate(w, h, iAlignUp(w, numDimensions), false, NULL, (float *)img.data);
    cudaImg.Download();

    float initBlur = 0.0f;
    float thresh = 5.0f;
    InitSiftData(siftData[i], 4096, true, true);
    ExtractSift(siftData[i], cudaImg, 5, initBlur, thresh, 0.0f);

    totalNumSIFT += siftData[i].numPts;
    img.release();
  }

  // Copy SIFT histograms into one contiguous block of memory
  float *siftHistograms = new float[numDimensions * totalNumSIFT];
  int counter = 0;
  for (int i = 0; i < numTrainingImages; i++) {
    for (int j = 0; j < siftData[i].numPts; j++) {
      memcpy(siftHistograms + counter, siftData[i].h_data[j].data, numDimensions * sizeof(float));
      counter++;
    }
  }

  cout << "Clustering SIFT training data into k means" << endl;

  kMeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
  vl_kmeans_set_algorithm(kMeans, VlKMeansANN);
  vl_kmeans_init_centers_with_rand_data (kMeans, siftHistograms, numDimensions, totalNumSIFT, numFeatures);
  vl_kmeans_set_max_num_iterations (kMeans, 500);
  features = (float *)vl_kmeans_get_centers(kMeans);

  cout << "Building vocabulary tree..." << endl;
  
  vl_size numTrees = 3;
  kdForest = vl_kdforest_new(VL_TYPE_FLOAT, numDimensions, numTrees, VlDistanceL2);
  vl_size maxNumComparisons = 500;
  vl_kdforest_set_max_num_comparisons(kdForest, maxNumComparisons);
  vl_kdforest_build(kdForest, totalNumSIFT, features);

  cout << "Computing inverse document frequency weights..." << endl;

  // First, compute number of images with a particular term (feature)
  for (int i = 0; i < numTrainingImages; i++) {
    float *histogram = quantize(siftData[i]);
    
    for (int j = 0; j < numFeatures; j++) {
      if (histogram[j] > 0) {
        weights[j]++;
      }
    }

    free(histogram);
  }

  // Then, compute the inverse frequency
  float numerator = log(numTrainingImages + 1);
  for (int i = 0; i < numFeatures; i++) {
    weights[i] = numerator / max(weights[i], 1.0);
  }

  cout << "Done!" << endl;

  free(siftHistograms);
  for (int i = 0; i < numTrainingImages; i++) {
    FreeSiftData(siftData[i]);
  }
}

float *cuBoF::quantize(SiftData siftData) {
  vl_size numNeighbors = 1;
  VlKDForestNeighbor kdForestNeighbor;

  float *histogram = new float[numFeatures]();
  for (int i = 0; i < siftData.numPts; i++) {
    float *query = siftData.h_data[i].data;
    vl_kdforest_query(kdForest, &kdForestNeighbor, numNeighbors, query);
    histogram[kdForestNeighbor.index]++;
  }

  normalize(histogram);

  return histogram;
}

void cuBoF::normalize(float *histogram) {
  float squaresum = 0;
  for (int i = 0; i < numFeatures; i++) {
    histogram[i] *= weights[i];
    squaresum += histogram[i] * histogram[i];
  }

  for (int i = 0; i < numFeatures; i++) {
    histogram[i] /= sqrt(squaresum);
  }
}

cuBoF::~cuBoF() {
  vl_kmeans_delete(kMeans);
  vl_kdforest_delete(kdForest);
}

