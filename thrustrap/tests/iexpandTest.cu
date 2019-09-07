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
#include "iexpand.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include <ostream>

int main(void)
{
  int counts[] = {3,5,2,0,1,3,4,2,4};

  size_t input_size  = sizeof(counts) / sizeof(int);
  size_t output_size = thrust::reduce(counts, counts + input_size); // sum of count values

  // copy inputs to device
  thrust::device_vector<int> d_counts(counts, counts + input_size);
  thrust::device_vector<int> d_output(output_size, 0);

  // expand 0:N-1 indices of counts according to count values
  iexpand(d_counts.begin(), 
          d_counts.end(), 
          d_output.begin(),
          d_output.end()
         );

  std::cout << "iExpanding indices according to counts" << std::endl;
  print(" counts ", d_counts);
  print(" output ", d_output);

  return 0;
}
