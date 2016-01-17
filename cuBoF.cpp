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
  fread((void *)&maxNumIterations, sizeof(uint32_t), 1, fp);
  fread((void *)&numNeighbors, sizeof(uint32_t), 1, fp);
  
  fclose(fp);

  fprintf(stderr, "Max num threads %llu\n", vl_get_max_threads());
  vl_set_num_threads(vl_get_max_threads());

  buildFeatureTree();
}

cuBoF::cuBoF(int numFeatures, int numTrainingImages) {
  this->numFeatures = numFeatures;
  this->numTrainingImages = numTrainingImages;

  numDimensions = 128;

  features = new float[numDimensions * numFeatures];
  weights = new float[numFeatures]();

  maxNumComparisons = 500;
  maxNumIterations = 100;
  numTrees = 3;
  numNeighbors = 1;

  fprintf(stderr, "Max num threads %llu\n", vl_get_max_threads());
  vl_set_num_threads(vl_get_max_threads());
}

cuBoF::~cuBoF() {
  free(weights);
  free(features);
  vl_kdforest_delete(kdForest);
}

void cuBoF::train(float *imgData, int w, int h) {
  vector<SiftData *> siftData;

  int totalNumSIFT = 0;

  for (int i = 0; i < numTrainingImages; i++) {
    cout << "Extracting SIFT for image " << i << " of " << numTrainingImages << endl;
    SiftData *currSiftData = extractFeaturesFromImage(imgData + i * w * h, w, h);
    siftData.push_back(currSiftData);
    totalNumSIFT += siftData[i]->numPts;
  }

  train(siftData, totalNumSIFT);
}

void cuBoF::train(vector<SiftData *> &siftData, int totalNumSIFT) {
  float *siftHistograms = new float[numDimensions * totalNumSIFT];

  // Copy SIFT histograms into one contiguous block of memory
  int counter = 0;
  for (int i = 0; i < numTrainingImages; i++) {
    for (int j = 0; j < siftData[i]->numPts; j++) {
      memcpy(siftHistograms + counter * numDimensions, siftData[i]->h_data[j].data, numDimensions * sizeof(float));
      // for (int k = 0; k < numDimensions; k++) {
      //   if (siftHistograms[counter * numDimensions + k] != siftData[i]->h_data[j].data[k]) {
      //     cout << siftHistograms[counter * numDimensions + k] << " " << siftData[i]->h_data[j].data[k] << endl;
      //   }
      // }
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
    FreeSiftData(*siftData[i]);
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
  VlKDForestNeighbor kdForestNeighbor;

  // cout << "Num sift keypoints: " << siftData->numPts << endl;
  
  for (int i = 0; i < siftData->numPts; i++) {
    float *query = siftData->h_data[i].data;
    vl_kdforest_query(kdForest, &kdForestNeighbor, numNeighbors, query);
    histogram[kdForestNeighbor.index]++;
  }

  // cout << "Quantized:" << endl;
  // for (int i = 0; i < numFeatures; i++) {
  //   cout << histogram[i] << " ";
  // }
  // cout << endl;
}

void cuBoF::weight(float *histogram) {
  for (int i = 0; i < numFeatures; i++) {
    histogram[i] *= weights[i];
  }
  // cout << "Weighted:" << endl;
  // for (int i = 0; i < numFeatures; i++) {
  //   cout << histogram[i] << " ";
  // }
  // cout << endl;
}

void cuBoF::normalize(float *histogram) {
  float squaresum = 0;
  for (int i = 0; i < numFeatures; i++) {
    squaresum += histogram[i] * histogram[i];
  }

  float sum = sqrt(squaresum);

  for (int i = 0; i < numFeatures; i++) {
    histogram[i] /= sum;
  }
  // cout << "Normalized:" << endl;
  // for (int i = 0; i < numFeatures; i++) {
  //   cout << histogram[i] << " ";
  // }
  // cout << endl;

  // float sum = 0;
  // for (int i = 0; i < numFeatures; i++) {
  //   sum += histogram[i] * histogram[i];
  // }
  // cout << "Total: " << sum << endl;
}

// Essentially a subsetted port of annkmeans.m from vlfeat
void cuBoF::clusterFeatures(int numPts, float *histograms) {

  // Total 'energy' of our optimization
  vector<float> energy(maxNumIterations, 0);

  // Copy a subset of the SIFT histograms to work with (15 taken from vlfeat
  // annkmeans.m). We could use entire set of histograms as well.
  int numData = numFeatures * 15;
  float *subset = getRowSubset(numPts, numData, numDimensions, histograms);

  // Get random subset as initial kmeans centers
  float *centers = getRowSubset(numData, numFeatures, numDimensions, subset);
  vector<float> dist(numData, FLT_MAX);
  vector<int> assign(numData, 0);
  vector<float> mass(numFeatures, 0);

  for (int i = 0; i < maxNumIterations; i++) {
    cerr << "Processing iteration " << i << endl;

    VlKDForest *forest = vl_kdforest_new(VL_TYPE_FLOAT, numDimensions, numTrees, VlDistanceL2);
    vl_kdforest_set_max_num_comparisons(forest, maxNumComparisons);
    vl_kdforest_build(forest, numFeatures, centers);

    // cerr << "Finding neighbors" << endl;
    for (int w = 0; w < numData; w++) {
      VlKDForestNeighbor neighbor;
      float *query = subset + w * numDimensions;
      vl_kdforest_query(forest, &neighbor, numNeighbors, query);
      if (neighbor.distance < dist[w]) {
        dist[w] = neighbor.distance;
        assign[w] = neighbor.index;
      }
    }

    // Get the number of times each center was closest to a SIFT histogram
    // cerr << "Calculating mass" << endl;
    for (int j = 0; j < numFeatures; j++) {
      mass[j] = 0;
      for (int w = 0; w < numData; w++) {
        if (assign[w] == j) {
          mass[j]++;
        }
      }
    }

    // Update the centers with an average of the subset SIFT histograms that
    // were closest to it
    // cerr << "Zeroing out affected centers" << endl;
    for (int w = 0; w < numData; w++) {
      for (int k = 0; k < numDimensions; k++) {
        centers[assign[w] * numDimensions + k] = 0;
      }
    }

    // cerr << "Recomputing centers" << endl;
    for (int w = 0; w < numData; w++) {
      for (int k = 0; k < numDimensions; k++) {
        centers[assign[w] * numDimensions + k] += subset[w * numDimensions + k];
      }
    }

    // Compute the energy of the current optimization
    // cerr << "Computing energy" << endl;
    for (int w = 0; w < numData; w++) {
      energy[i] += dist[w];
    }

    // Find the centers that got no love
    // cerr << "Finding centers to replace" << endl;
    vector<int> centersToReplace;
    for (int j = 0; j < numFeatures; j++) {
      if (mass[j] == 0) {
        centersToReplace.push_back(j);
      }
    }

    // cerr << "Replacing centers" << endl;
    // Replace them
    vector<int> centerReplacements = getRandomIntVector(0, numData - 1, centersToReplace.size());
    for (int j = 0; j < centersToReplace.size(); j++) {
      for (int k = 0; k < numDimensions; k++) {
        centers[centersToReplace[j] * numDimensions + k] = subset[centerReplacements[j] * numDimensions + k];
      }
    }

    // If we haven't improved our total distance metric by some threshold,
    // break out of the loop
    if (i > 0 && energy[i] > 0.999 * energy[i - 1]) {
      break;
    }

    vl_kdforest_delete(forest);
  }

  free(subset);
  features = centers;

  // VlKMeans *kMeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
  // vl_kmeans_set_algorithm(kMeans, VlKMeansANN);
  // vl_kmeans_set_initialization(kMeans, VlKMeansRandomSelection);
  // vl_kmeans_cluster(kMeans, histograms, numDimensions, numPts, numFeatures);
  // // vl_kmeans_init_centers_with_rand_data(kMeans, features, numDimensions, numPts, numFeatures);
  // // vl_kmeans_set_centers(kMeans, histograms, numDimensions, numFeatures);
  // vl_kmeans_set_max_num_iterations(kMeans, maxNumIterations);
  // // vl_kmeans_refine_centers(kMeans, histograms, numPts);
  // memcpy(features, (float *)vl_kmeans_get_centers(kMeans), numDimensions * numFeatures * sizeof(float));
  // vl_kmeans_delete(kMeans);
}

void cuBoF::buildFeatureTree() {
  kdForest = vl_kdforest_new(VL_TYPE_FLOAT, numDimensions, numTrees, VlDistanceL2);
  vl_kdforest_set_max_num_comparisons(kdForest, maxNumComparisons);
  vl_kdforest_build(kdForest, numFeatures, features);
}

void cuBoF::computeWeights(vector<SiftData *> siftData) {
  float *histogram = new float[numFeatures]();

  // First, compute number of images with a particular term (feature)
  for (int i = 0; i < numTrainingImages; i++) {

    // Get histogram of features that occur in image i
    quantize(siftData[i], histogram);
    
    // Increment number of images that contain each feature
    for (int j = 0; j < numFeatures; j++) {

      // This is sort of confusing; this isn't IDF weight yet. At this
      // point, it's just the number of images so far that contain the
      // feature
      if (histogram[j] > 0) {
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
    weights[i] = numerator / max(weights[i], 0.0001f);
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
  fwrite(&maxNumIterations, sizeof(uint32_t), 1, fp);
  fwrite(&numNeighbors, sizeof(uint32_t), 1, fp);

  fclose(fp);
}