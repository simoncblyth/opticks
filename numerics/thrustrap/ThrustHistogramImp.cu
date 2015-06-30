
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "ThrustHistogramImp.h"

#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iomanip>
#include <iterator>

#ifdef DEBUG
#include "print_vector.h"
#endif


void sparse_histogram_imp(const thrust::device_vector<unsigned long long>& input,
                                thrust::device_vector<unsigned long long>& histogram_values,
                                thrust::device_vector<int>& histogram_counts,
                                thrust::device_vector<int>& histogram_index)
{

  thrust::device_vector<unsigned long long> data(input);  // copying input to avoid sorting it 

#ifdef DEBUG
  unsigned int stride = 1000 ; 
  print_vector_strided("initial data", data, true, stride );
#endif

  thrust::sort(data.begin(), data.end());
  
#ifdef DEBUG
  print_vector_strided("sorted data", data, true, stride );
#endif

  int num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                             data.begin() + 1,
                                             int(1),
                                             thrust::plus<int>(),
                                             thrust::not_equal_to<unsigned long long>());

  // resize histogram storage
  histogram_values.resize(num_bins);
  histogram_counts.resize(num_bins);
  histogram_index.resize(num_bins);
  
  // compact find the end of each bin of values
  thrust::reduce_by_key(data.begin(), data.end(),
                        thrust::constant_iterator<int>(1),
                        histogram_values.begin(),
                        histogram_counts.begin());
  

#ifdef DEBUG
  print_vector("histogram values", histogram_values, true);
  print_vector("histogram counts", histogram_counts, false, true );
#endif

  
  thrust::sort_by_key( histogram_counts.begin(), histogram_counts.end(), histogram_values.begin());
                
#ifdef DEBUG
  print_vector("histogram values (sorted by counts)", histogram_values, true);
  print_vector("histogram counts (sorted by counts)", histogram_counts, false, true);
#endif


}


template <typename T>
void direct_dump(T* devptr, unsigned int numElements)
{
    // dumping directly from device memory  : not efficient, but convenient for debugging
 
    thrust::device_ptr<T>  ptr = thrust::device_pointer_cast(devptr) ;

    std::cout << std::hex ; 

    if(numElements < 100 )
    {
        thrust::copy(ptr, ptr+numElements, std::ostream_iterator<T>(std::cout, "\n")); 
    } 
    else
    {
        thrust::copy(ptr, ptr+10, std::ostream_iterator<T>(std::cout, "\n")); 
        std::cout << "..." << std::endl ; 
        thrust::copy(ptr + numElements - 10, ptr + numElements, std::ostream_iterator<T>(std::cout, "\n"));  
    } 
    std::cout << std::endl ; 
}





