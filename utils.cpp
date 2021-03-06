#include "utils.h"

float *loadTrainingDataFromList(const char *trainingDataList, int *numTrainingImages, int *w, int *h) {

  string line;
  ifstream list(trainingDataList);

  if (list.is_open()) {
    list >> *numTrainingImages;
    list >> *w;
    list >> *h;

    cout << "Found " << *numTrainingImages << " images with width " << *w << " and height " << *h << endl;  
    cout << "Need to allocate " << *w * *h * *numTrainingImages << " bytes of memory" << endl;

    float *imgData = (float *)malloc(*w * *h * *numTrainingImages * sizeof(float));

    int counter = 0;

    // Finish off empty line
    getline(list, line);

    while (getline(list, line)) {
      Mat img;
      imread(line, 0).convertTo(img, CV_32FC1);

      memcpy(imgData + counter * *w * *h, (float *)img.data, *w * *h * sizeof(float));

      counter++;
      img.release();
    }

    list.close();
    return imgData;
  }

  return NULL;
}

SiftData *extractFeaturesFromImage(float *imgData, int w, int h) {
  SiftData *siftData = new SiftData();

  InitCuda(0);
  cuImage cudaImg;
  cudaImg.Allocate(w, h, iAlignUp(w, 128), false, NULL, imgData);
  cudaImg.Download();

  float initBlur = 0.0f;
  float thresh = 5.0f;
  InitSiftData(*siftData, 4096, true, true);
  ExtractSift(*siftData, cudaImg, 5, initBlur, thresh, 0.0f);

  cout << "Extracted " << siftData->numPts << " key points" << endl;

  return siftData;
}

float dot(float *hist1, float *hist2, int numBins) {
  float sum = 0;

  for (int i = 0; i < numBins; i++) {
    sum += hist1[i] * hist2[i];
  }

  return sum;
}

float intersect(float *hist1, float *hist2, int numBins) {
  float sum = 0;
  for (int i = 0; i < numBins; i++) {
    sum += min(hist1[i], hist2[i]);
  }

  return sum;
}

vector<int> getRandomIntVector(int lower, int upper, int length, bool replace) {
  cerr << "Generating random vector from " << lower << " to " << upper << " with length " << length << endl;
  random_device rnd_device;
  mt19937 mersenne_engine(rnd_device());
  uniform_int_distribution<int> dist(lower, upper);

  vector<int> result;

  if (replace) {
    result.resize(length);
    auto gen = bind(dist, mersenne_engine);
    generate(begin(result), end(result), gen);
  } else {
    for (int i = lower; i <= upper; i++) {
      result.push_back(i);
    }
    random_shuffle(result.begin(), result.end());
    result.resize(length);
  }

  return result;
}

float *getRowSubset(int total, int subset, int dim, float *data) {
  vector<int> indices = getRandomIntVector(0, total - 1, subset, false);
  float *result = new float[subset * dim];

  for (int i = 0; i < subset; i++) {
    memcpy(result + i * dim, data + indices[i] * dim, sizeof(float) * dim);
  }

  return result;
}

Mat combineMatchedImages(Mat img1, Mat img2) {
  Mat img3(img1.size().height, img1.size().width + img2.size().width, CV_32FC1);
  Mat left(img3, Rect(0, 0, img1.size().width, img1.size().height));
  img1.copyTo(left);
  Mat right(img3, Rect(img1.size().width, 0, img2.size().width, img2.size().height));
  img2.copyTo(right);

  return img3;
}

void writeArrayToFile(string path, float *data, int rows, int cols) {
  FILE *fp = fopen(path.c_str(), "wb");

  fwrite(&rows, sizeof(uint32_t), 1, fp);
  fwrite(&cols, sizeof(uint32_t), 1, fp);
  fwrite(data, sizeof(float), rows * cols, fp);

  fclose(fp);
}

/* 
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  File:    nms.cpp
 *  Author:  Hilton Bristow
 *  Created: Jul 19, 2012
 */

 /*! @brief suppress non-maximal values
 *
 * nonMaximaSuppression produces a mask (dst) such that every non-zero
 * value of the mask corresponds to a local maxima of src. The criteria
 * for local maxima is as follows:
 *
 *  For every possible (sz x sz) region within src, an element is a
 *  local maxima of src iff it is strictly greater than all other elements
 *  of windows which intersect the given element
 *
 * Intuitively, this means that all maxima must be at least sz+1 pixels
 * apart, though the spacing may be greater
 *
 * A gradient image or a constant image has no local maxima by the definition
 * given above
 *
 * The method is derived from the following paper:
 * A. Neubeck and L. Van Gool. "Efficient Non-Maximum Suppression," ICPR 2006
 *
 * Example:
 * \code
 *  // create a random test image
 *  Mat random(Size(2000,2000), DataType<float>::type);
 *  randn(random, 1, 1);
 *
 *  // only look for local maxima above the value of 1
 *  Mat mask = (random > 1);
 *
 *  // find the local maxima with a window of 50
 *  Mat maxima;
 *  nonMaximaSuppression(random, 50, maxima, mask);
 *
 *  // optionally set all non-maxima to zero
 *  random.setTo(0, maxima == 0);
 * \endcode
 *
 * @param src the input image/matrix, of any valid cv type
 * @param sz the size of the window
 * @param dst the mask of type CV_8U, where non-zero elements correspond to
 * local maxima of the src
 * @param mask an input mask to skip particular elements
 */
void nonMaximaSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask) {

  // initialise the block mask and destination
  const int M = src.rows;
  const int N = src.cols;
  const bool masked = !mask.empty();
  Mat block = 255*Mat_<uint8_t>::ones(Size(2*sz+1,2*sz+1));
  dst = Mat_<uint8_t>::zeros(src.size());

  // iterate over image blocks
  for (int m = 0; m < M; m+=sz+1) {
    for (int n = 0; n < N; n+=sz+1) {
      Point  ijmax;
      double vcmax, vnmax;

      // get the maximal candidate within the block
      Range ic(m, min(m+sz+1,M));
      Range jc(n, min(n+sz+1,N));
      minMaxLoc(src(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic,jc) : noArray());
      Point cc = ijmax + Point(jc.start,ic.start);

      // search the neighbours centered around the candidate for the true maxima
      Range in(max(cc.y-sz,0), min(cc.y+sz+1,M));
      Range jn(max(cc.x-sz,0), min(cc.x+sz+1,N));

      // mask out the block whose maxima we already know
      Mat_<uint8_t> blockmask;
      block(Range(0,in.size()), Range(0,jn.size())).copyTo(blockmask);
      Range iis(ic.start-in.start, min(ic.start-in.start+sz+1, in.size()));
      Range jis(jc.start-jn.start, min(jc.start-jn.start+sz+1, jn.size()));
      blockmask(iis, jis) = Mat_<uint8_t>::zeros(Size(jis.size(),iis.size()));

      minMaxLoc(src(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in,jn).mul(blockmask) : blockmask);
      Point cn = ijmax + Point(jn.start, in.start);

      // if the block centre is also the neighbour centre, then it's a local maxima
      if (vcmax > vnmax) {
        dst.at<uint8_t>(cc.y, cc.x) = 255;
      }
    }
  }
}

// % Find best loop closure candidates for each image via non-maximum
// % suppression
// function scoreNMS = nmsMatrix(score, radius)

// % Create result matrix
// scoreNMS = zeros(size(score));

// % Iterate over columns in score matrix
// for i = 1:size(score, 2)
    
//     % Suppress non maximums in a given column (i.e., similarity scores to
//     % other images)
//     pickArray = nmsMatrixOneD(score(:, i), radius);
    
//     % MATLAB is column major order, so we grab the indices specified by
//     % pickArray and transfer over the scores
//     scoreNMS((i - 1) * size(score, 1) + pickArray) = score((i - 1) * size(score, 1) + pickArray);
// end

// % Return a sparse matrix to save space
// scoreNMS = sparse(scoreNMS);
// end

// % Find the maximum element in an array and set all elements within radius
// % indices to 0. Repeat until all elements except maximums are equal to
// % zero.
// function pickArray = nmsMatrixOneD(scoreArray, radius)

//     % Get sort indices of the score column
//     [~, ind] = sort(scoreArray);
    
//     % Create result array
//     pickArray = zeros(1, length(scoreArray));
    
//     % Count number of maxima we find
//     cnt = 1;
    
//     % Loop over score array
//     while sum(scoreArray > 0)
        
//         % Find index of current maximum
//         pickInd = ind(end);
        
//         % Store the index of the maximum
//         pickArray(cnt) = pickInd;
        
//         % Zero out any part of the array that isn't zero
//         scoreArray([max(pickInd - radius, 1):min(length(scoreArray), pickInd + radius)]) = 0;
        
//         % Remove corresponding indices that are now point to zeros
//         ind(ind < (pickInd + radius + 1) & ind > (pickInd - radius - 1)) = [];
        
//         % Increment number of maxima found
//         cnt = cnt +1;
//     end
    
//     % Truncate to the number of maxima found
//     pickArray = pickArray(1:cnt-1);
// end