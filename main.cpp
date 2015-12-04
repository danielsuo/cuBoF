#include <iostream>

extern "C" {
  #include <vl/generic.h>
  #include <vl/kmeans.h>
}

using namespace std;

int main(int argc, char **argv) {

	VL_PRINT("Hello world!\n");

	VlKMeans *kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
	return 0;
}