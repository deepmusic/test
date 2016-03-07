#include "layer.h"
#include <stdio.h>

// total number of elements in a tensor
int flatten_size(const Tensor* const tensor)
{
  int total_size = 0;
  for (int n = 0; n < tensor->num_items; ++n) {
    int size = 1;
    for (int d = 0; d < tensor->ndim; ++d) {
      size *= tensor->shape[n][d];
    }
    total_size += size;
  }
  return total_size;
}

// load binary data from file
//   if "data" is NULL, allocate memory & load data & return pointer
//   otherwise, load data to "data"
real* load_data(const char* const filename,
                int* const ndim,
                int* const shape,
                real* data)
{
  FILE* fp = fopen(filename, "rb");

  // load data shape
  {
    if ((int)fread(ndim, sizeof(int), 1, fp) < 1) {
      printf("Error while reading ndim from %s\n", filename);
    }
    if ((int)fread(shape, sizeof(int), *ndim, fp) != *ndim) {
      printf("Error while reading shape from %s\n", filename);
    }
  }

  // compute total number of elements
  {
    const int ndim_ = *ndim;
    int count = 1;
    for (int i = 0; i < ndim_; ++i) {
      count *= shape[i];
    }
    shape[ndim_] = count;
  }

  // memory allocation & load data
  {
    const int count = shape[*ndim];
    if (data == NULL) {
      data = (real*)malloc(count * sizeof(real));
    }
    if ((int)fread(data, sizeof(real), count, fp) != count) {
      printf("Error while reading data from %s\n", filename);
    }

    // file close & return data
    fclose(fp);
    return data;
  }
}
