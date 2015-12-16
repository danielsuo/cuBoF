#include "cuBoF.h"

cuBoF::cuBoF(const char *path) {
  fprintf(stderr, "Reading BoF file from %s\n", path);

  FILE *fp = fopen(path, "rb");

  fread((void *)&numDimensions, sizeof(uint32_t), 1, fp);
  fread((void *)&numFeatures, sizeof(uint32_t), 1, fp);

  features = new float[numDimensions * numFeatures];
  weights = new float[numFeatures]();
  
  fread((void *)features, sizeof(float), numDimensions * numFeatures, fp);
  fread((void *)weights, sizeof(float), numFeatures, fp);
  fread((void *)&numTrees, sizeof(uint32_t), 1, fp);
  fread((void *)&maxNumComparisons, sizeof(uint32_t), 1, fp);
  
  fclose(fp);

  buildFeatureTree();
}

cuBoF::cuBoF(int numFeatures, int numTrainingImages) {
  this->numFeatures = numFeatures;
  this->numTrainingImages = numTrainingImages;

  numDimensions = 128;

  features = new float[numDimensions * numFeatures];
  weights = new float[numFeatures]();

  maxNumComparisons = 500;
  numTrees = 3;
}

cuBoF::~cuBoF() {
  vl_kdforest_delete(kdForest);
}

void cuBoF::train(float *imgData, int w, int h) {
  SiftData *siftData = new SiftData[numTrainingImages];

  int totalNumSIFT = 0;

  for (int i = 0; i < numTrainingImages; i++) {
    cout << "Extracting SIFT for image " << i << " of " << numTrainingImages << endl;
    extractFeaturesFromImage(imgData + i * w * h, w, h, siftData[i]);
    totalNumSIFT += siftData[i].numPts;
  }

  float *siftHistograms = new float[numDimensions * totalNumSIFT];

  // Copy SIFT histograms into one contiguous block of memory
  int counter = 0;
  for (int i = 0; i < numTrainingImages; i++) {
    for (int j = 0; j < siftData[i].numPts; j++) {
      memcpy(siftHistograms + counter, siftData[i].h_data[j].data, numDimensions * sizeof(float));
      counter++;
    }
  }

  cout << "Clustering SIFT training data into k means" << endl;
  clusterFeatures(totalNumSIFT, siftHistograms);

  cout << "Building vocabulary tree..." << endl;
  buildFeatureTree();

  cout << "Computing inverse document frequency weights..." << endl;
  computeWeights(siftData);

  cout << "Done!" << endl;

  free(siftHistograms);
  for (int i = 0; i < numTrainingImages; i++) {
    FreeSiftData(siftData[i]);
  }
}

float *cuBoF::vectorize(SiftData *siftData) {
  float *histogram = new float[numFeatures]();

  quantize(siftData, histogram);
  weight(histogram);
  normalize(histogram);

  return histogram;
}

void cuBoF::quantize(SiftData *siftData, float *histogram) {
  vl_size numNeighbors = 1;
  VlKDForestNeighbor kdForestNeighbor;
  
  for (int i = 0; i < siftData->numPts; i++) {
    float *query = siftData->h_data[i].data;
    vl_kdforest_query(kdForest, &kdForestNeighbor, numNeighbors, query);
    histogram[kdForestNeighbor.index]++;
  }
}

void cuBoF::weight(float *histogram) {
  for (int i = 0; i < numFeatures; i++) {
    histogram[i] *= weights[i];
  }
}

void cuBoF::normalize(float *histogram) {
  float squaresum = 0;
  for (int i = 0; i < numFeatures; i++) {
    squaresum += histogram[i] * histogram[i];
  }

  for (int i = 0; i < numFeatures; i++) {
    histogram[i] /= sqrt(squaresum);
  }
}

void cuBoF::extractFeaturesFromImage(float *imgData, int w, int h, SiftData &siftData) {
  InitCuda(0);
  CudaImage cudaImg;
  cudaImg.Allocate(w, h, iAlignUp(w, numDimensions), false, NULL, imgData);
  cudaImg.Download();

  float initBlur = 0.0f;
  float thresh = 5.0f;
  InitSiftData(siftData, 4096, true, true);
  ExtractSift(siftData, cudaImg, 5, initBlur, thresh, 0.0f);
}

void cuBoF::clusterFeatures(int numPts, float *histograms) {
  VlKMeans *kMeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
  vl_kmeans_set_algorithm(kMeans, VlKMeansANN);
  vl_kmeans_init_centers_with_rand_data (kMeans, histograms, numDimensions, numPts, numFeatures);
  vl_kmeans_set_max_num_iterations (kMeans, 500);
  memcpy(features, (float *)vl_kmeans_get_centers(kMeans), numDimensions * numFeatures * sizeof(float));
  vl_kmeans_delete(kMeans);
}

void cuBoF::buildFeatureTree() {
  kdForest = vl_kdforest_new(VL_TYPE_FLOAT, numDimensions, numTrees, VlDistanceL2);
  vl_kdforest_set_max_num_comparisons(kdForest, maxNumComparisons);
  vl_kdforest_build(kdForest, numFeatures, features);
}

void cuBoF::computeWeights(SiftData *siftData) {
  float *histogram = new float[numFeatures]();

  // First, compute number of images with a particular term (feature)
  for (int i = 0; i < numTrainingImages; i++) {

    // Get histogram of features that occur in image i
    quantize(siftData + i, histogram);
    
    // Increment number of images that contain each feature
    for (int j = 0; j < numFeatures; j++) {
      if (histogram[j] > 0) {
        // This is sort of confusing; this isn't IDF weight yet. At this
        // point, it's just the number of images so far that contain the
        // feature
        weights[j]++;
      }

      // Reset histogram variable
      histogram[j] = 0;
    }
  }

  free(histogram);

  // Then, compute the inverse frequency
  float numerator = log(numTrainingImages + 1);
  for (int i = 0; i < numFeatures; i++) {
    weights[i] = numerator / max(weights[i], 1.0);
  }
}

void cuBoF::save(const char *path) {
  fprintf(stderr, "Saving BoF to file %s\n", path);

  FILE *fp = fopen(path, "wb");

  fwrite(&numDimensions, sizeof(uint32_t), 1, fp);
  fwrite(&numFeatures, sizeof(uint32_t), 1, fp);
  fwrite(features, sizeof(float), numDimensions * numFeatures, fp);
  fwrite(weights, sizeof(float), numFeatures, fp);
  fwrite(&numTrees, sizeof(uint32_t), 1, fp);
  fwrite(&maxNumComparisons, sizeof(uint32_t), 1, fp);

  fclose(fp);
}