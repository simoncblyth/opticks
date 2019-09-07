/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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
