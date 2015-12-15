#include "cuBoF.h"

cuBoF::cuBoF(int numFeatures, int numTrainingImages) {
  this->numFeatures = numFeatures;
  this->numTrainingImages = numTrainingImages;

  features = new float[128 * numFeatures];
  weights = new float[numFeatures];
}

void cuBoF::train(char **paths) {
  // Initialize Mat to hold Sift keypoints. This way of allocating data is
  // really inefficient, but this example is for offline training
  cv::Mat data = cv::Mat::zeros(0, 128, CV_32FC1);
  float *numSiftPoints = new float[numTrainingImages];

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
    cudaImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, (float *)img.data);
    cudaImg.Download();

    SiftData siftData;
    float initBlur = 0.0f;
    float thresh = 5.0f;
    InitSiftData(siftData, 4096, true, true);
    ExtractSift(siftData, cudaImg, 5, initBlur, thresh, 0.0f);

    for (int j = 0; j < siftData.numPts; j++) {
      cv::Mat hist = cv::Mat(1, 128, CV_32FC1, siftData.h_data[j].data);
      cout << hist << endl << endl;;
      data.push_back(hist);
    }

    numSiftPoints[i] = siftData.numPts;

    FreeSiftData(siftData);
    img.release();
  }

  cout << "Clustering SIFT training data into k means" << endl;

  kMeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
  vl_kmeans_set_algorithm(kMeans, VlKMeansANN);
  vl_kmeans_init_centers_with_rand_data (kMeans, data.data, data.cols, data.rows, numFeatures);
  vl_kmeans_set_max_num_iterations (kMeans, 100);
  features = (float *)vl_kmeans_get_centers(kMeans);

  cout << "Building vocabulary tree..." << endl;
  
  vl_size numTrees = 3;
  kdForest = vl_kdforest_new(VL_TYPE_FLOAT, data.cols, numTrees, VlDistanceL2);
  vl_size maxNumComparisons = 500;
  vl_kdforest_set_max_num_comparisons(kdForest, maxNumComparisons);
  vl_kdforest_build(kdForest, data.rows, features);

  cout << "Done!" << endl;

  data.release();
}

void cuBoF::quantize(int numSiftPoints, float *siftPointHistograms) {
  
}

cuBoF::~cuBoF() {
  vl_kmeans_delete(kMeans);
  vl_kdforest_delete(kdForest);
}

