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
/** 

iexpand.h
===========

Adapted from  /usr/local/env/numerics/thrust/examples/expand.cu 

Expand an input sequence of counts by replicating indices of each element the number
of times specified by the count values. 

The element counts are assumed to be non-negative integers.

Note that the length of the output is equal 
to the sum of the input counts.

For example::

    iexpand([2,2,2]) -> [0,0,1,1,2,2]  2*0, 2*1, 2*2
    iexpand([3,0,1]) -> [0,0,0,2]      3*0, 0*1, 1*2
    iexpand([1,3,2]) -> [0,1,1,1,2,2]  1*0, 3*1, 2*2 

NB the output device must be zeroed prior to calling iexpand. 
This is because the iexpand is implemented ending with an inclusive_scan 
to fill in the non-transition values which relies on initial zeroing.


A more specific example:

Every optical photon generating genstep (Cerenkov or scintillation) 
specifies the number of photons it will generate.
Applying iexpand to the genstep photon counts produces
an array of genstep indices that is stored into the seed buffer
and provides a reference back to the genstep that produced it.
The seed values are used to translate from a photon index to a 
genstep index. 

**/


#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <iostream>

#ifdef DEBUG
#include <cassert>
#include <iterator>
template <typename Vector>
void print(const std::string& s, const Vector& v)
{
  typedef typename Vector::value_type T;

  std::cout << s;
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}
#endif


template <typename InputIterator,
          typename OutputIterator>
void iexpand(InputIterator  counts_first,
             InputIterator  counts_last,
             OutputIterator output_first,
             OutputIterator output_last)
{
  typedef typename thrust::iterator_difference<InputIterator>::type difference_type;
  
  difference_type counts_size = thrust::distance(counts_first, counts_last);  // eg number of gensteps
  difference_type output_size = thrust::distance(output_first, output_last);  // eg number of photon "seeds" : back referencing genstep index 

#ifdef DEBUG
  std::cout << "iexpand " 
            << " counts_size " << counts_size  
            << " output_size " << output_size  
            << std::endl ; 
#endif


  thrust::device_vector<difference_type> output_offsets(counts_size, 0);

  thrust::exclusive_scan(counts_first, counts_last, output_offsets.begin());  
#ifdef DEBUG
  print(
     " scan the counts to obtain output offsets for each input element \n"
     " exclusive_scan of input counts creating output_offsets of transitions \n"
     " exclusive_scan is a cumsum that excludes current value \n"
     " 1st result element always 0, last input value ignored  \n"
     " (output_offsets) \n"
   , output_offsets );

  difference_type output_size2 = thrust::reduce(counts_first, counts_last);    // sum of input counts 
  assert( output_size == output_size2 ); 
#endif

  // scatter indices into transition points of output 
  thrust::scatter_if
    (thrust::counting_iterator<difference_type>(0),
     thrust::counting_iterator<difference_type>(counts_size),
     output_offsets.begin(),
     counts_first,
     output_first); 

#ifdef DEBUG
  printf(
     " scatter the nonzero counts into their corresponding output positions \n"
     " scatter_if( first, last, map, stencil, output ) \n"
     "    conditionally copies elements from a source range (indices 0:N-1) into an output array according to a map \n"
     "    condition dictated by a stencil (input counts) which must be non-zero to be true \n"
     "    map provides indices of where to put the indice values in the output  \n"
   );
#endif

  thrust::inclusive_scan
    (output_first,
     output_last,
     output_first,
     thrust::maximum<difference_type>());

#ifdef DEBUG
  printf(
     " compute max-scan over the output indices, filling in the holes \n"
     " inclusive_scan is a cumsum that includes current value \n"
     " providing an binary operator (maximum) replaces the default of plus \n" 
     "   because the empties are init to 0 this will fill in the empties to the right \n"
   );
#endif


}



