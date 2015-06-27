#include "histogram_sorted.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iomanip>
#include <iterator>

// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

// sparse histogram using reduce_by_key
template <typename Vector1,
          typename Vector2,
          typename Vector3,
          typename Vector4>
void sparse_histogram(const Vector1& input,
                            Vector2& histogram_values,
                            Vector3& histogram_counts,
                            Vector4& histogram_index)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector3::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // print the initial data
  print_vector("initial data", data);

  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());
  
  // print the sorted data
  print_vector("sorted data", data);

  // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
  IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                             data.begin() + 1,
                                             IndexType(1),
                                             thrust::plus<IndexType>(),
                                             thrust::not_equal_to<ValueType>());

  // resize histogram storage
  histogram_values.resize(num_bins);
  histogram_counts.resize(num_bins);
  histogram_index.resize(num_bins);
  
  // compact find the end of each bin of values
  thrust::reduce_by_key(data.begin(), data.end(),
                        thrust::constant_iterator<IndexType>(1),
                        histogram_values.begin(),
                        histogram_counts.begin());
  
  // print the sparse histogram

  print_vector("histogram values", histogram_values);
  print_vector("histogram counts", histogram_counts);

  
  thrust::sort_by_key( histogram_counts.begin(), histogram_counts.end(), 
                       histogram_values.begin());
                

  thrust::sequence( histogram_index.begin(), histogram_index.end() );
     
  print_vector("histogram values (sorted by counts)", histogram_values);
  print_vector("histogram counts (sorted bt counts)", histogram_counts);
  print_vector("histogram index                    ", histogram_index );


  //  http://docs.thrust.googlecode.com/hg/group__replacing.html
  //
  // hmm need to replace the data values with indices 

}

int histogram_sorted(void)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 9);

  const int N = 40;
  const int S = 4;

  // generate random data on the host
  thrust::host_vector<int> input(N);
  for(int i = 0; i < N; i++)
  {
    int sum = 0;
    for (int j = 0; j < S; j++)
      sum += dist(rng);
    input[i] = sum / S;
  }

 
  // demonstrate sparse histogram method
  {
    std::cout << "Sparse Histogram" << std::endl;

    thrust::device_vector<int> histogram_values;
    thrust::device_vector<int> histogram_counts;
    thrust::device_vector<int> histogram_index;

    sparse_histogram(input, 
                     histogram_values, 
                     histogram_counts,
                     histogram_index
                    );
  }


  // Note: 
  // A dense histogram can be converted to a sparse histogram
  // using stream compaction (i.e. thrust::copy_if).
  // A sparse histogram can be expanded into a dense histogram
  // by initializing the dense histogram to zero (with thrust::fill)
  // and then scattering the histogram counts (with thrust::scatter).

  return 0;
}

