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

#pragma once
//
// adapted from  /usr/local/env/numerics/thrust/examples/expand.cu 
//
// This example demonstrates how to expand an input sequence by 
// replicating each element a variable number of times. For example,
//
//   expand([2,2,2],[A,B,C]) -> [A,A,B,B,C,C]
//   expand([3,0,1],[A,B,C]) -> [A,A,A,C]
//   expand([1,3,2],[A,B,C]) -> [A,B,B,B,C,C]
//
// The element counts are assumed to be non-negative integers
//
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>


#ifdef DEBUG
#include <iterator>
#include <iostream>
template <typename Vector>
void print(const std::string& s, const Vector& v)
{
  typedef typename Vector::value_type T;

  std::cout << s;
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}
#endif


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator>
OutputIterator expand(InputIterator1 first1,
                      InputIterator1 last1,
                      InputIterator2 first2,
                      OutputIterator output)
{
  typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;
  
  difference_type input_size  = thrust::distance(first1, last1);
  difference_type output_size = thrust::reduce(first1, last1);    // sum of input counts 

  thrust::device_vector<difference_type> output_offsets(input_size, 0);
  thrust::exclusive_scan(first1, last1, output_offsets.begin());  

#ifdef DEBUG
  print(
     " scan the counts to obtain output offsets for each input element \n"
     " exclusive_scan of input counts creating output_offsets of transitions \n"
     " exclusive_scan is a cumsum that excludes current value \n"
     " 1st result element always 0, last input value ignored  \n"
     " (output_offsets) \n"
   , output_offsets );
#endif

  thrust::device_vector<difference_type> output_indices(output_size, 0); 
  thrust::scatter_if
    (thrust::counting_iterator<difference_type>(0),
     thrust::counting_iterator<difference_type>(input_size),
     output_offsets.begin(),
     first1,
     output_indices.begin()); 

#ifdef DEBUG
  print(
     " scatter the nonzero counts into their corresponding output positions \n"
     " scatter_if   first, last, map, stencil, output \n"
     "    conditionally copies elements from a source range (indices 0:N-1) into an output array according to a map \n"
     "    condition dictated by a stencil (inputs counts) which must be non-zero to be true \n"
     "    map provides indices of where to put the input values \n"
     " (output_indices) \n"
   , output_indices );
#endif

  thrust::inclusive_scan
    (output_indices.begin(),
     output_indices.end(),
     output_indices.begin(),
     thrust::maximum<difference_type>());

#ifdef DEBUG
  print(
     " compute max-scan over the output indices, filling in the holes \n"
     " inclusive_scan is a cumsum that includes current value \n"
     " providing an binary operator (maximum) replaces the default of plus \n" 
     "   because the empties are init to 0 this will fill in the empties to the right \n"
     " (output_indices) \n"
   , output_indices );

#endif


  OutputIterator output_end = output; 
  thrust::advance(output_end, output_size);
  thrust::gather(output_indices.begin(),
                 output_indices.end(),
                 first2,
                 output);

#ifdef DEBUG
  print(
     " gather input values according to index array (output = first2[output_indices]) \n"
     "   gather ( map_first, map_last, input_first, result ) \n"
     "       copies from input to result according to the map \n"
     "  \n"
     " (output_indices) \n"
   , output_indices );
#endif

  // return output + output_size
  thrust::advance(output, output_size);
  return output;
}





