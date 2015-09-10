#define DEBUG 1
#include "expand.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <ostream>

int main(void)
{
  int counts[] = {3,5,2,0,1,3,4,2,4};
  int values[] = {1,2,3,4,5,6,7,8,9};

  size_t input_size  = sizeof(counts) / sizeof(int);
  size_t output_size = thrust::reduce(counts, counts + input_size); // sum of counts

  // copy inputs to device
  thrust::device_vector<int> d_counts(counts, counts + input_size);
  thrust::device_vector<int> d_values(values, values + input_size);
  thrust::device_vector<int> d_output(output_size);

  // expand values according to counts
  expand(d_counts.begin(), d_counts.end(),
         d_values.begin(),
         d_output.begin());

  std::cout << "Expanding values according to counts" << std::endl;
  print(" counts ", d_counts);
  print(" values ", d_values);
  print(" output ", d_output);

  return 0;
}
