#pragma once
//
// adapted from  /usr/local/env/numerics/thrust/examples/expand.cu 
//
// Expand an input sequence by replicating indices of each element the number
// of times specified by the sequence values. 
//
// For example:
//
//   iexpand([2,2,2]) -> [0,0,1,1,2,2]
//   iexpand([3,0,1]) -> [0,0,0,2]
//   iexpand([1,3,2]) -> [0,1,1,1,2,2]
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
          typename OutputIterator>
void iexpand(InputIterator1 first1,
                      InputIterator1 last1,
                      OutputIterator output_first,
                      OutputIterator output_last)
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

  // scatter indices into transition points of output 
  thrust::scatter_if
    (thrust::counting_iterator<difference_type>(0),
     thrust::counting_iterator<difference_type>(input_size),
     output_offsets.begin(),
     first1,
     output_first); 

  // inclusive cum"sum" with max rather than sum, fills to the right 
  thrust::inclusive_scan
    (output_first,
     output_last,
     output_first,
     thrust::maximum<difference_type>());

}



