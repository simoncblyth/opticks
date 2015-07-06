#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "ThrustHistogramImp.h"

#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "strided_range.h"

#include <iostream>
#include <iomanip>
#include <iterator>

#ifdef DEBUG
#include "print_vector.h"
#endif



template <typename T>
void sparse_histogram_imp(const thrust::device_vector<T>& sequence,
                                unsigned int sequence_offset,   
                                unsigned int sequence_itemsize,   
                                thrust::device_vector<T>& histogram_values,
                                thrust::device_vector<int>& histogram_counts)
{
    // equivalent of my NumPy function count_unique
    // by filling a sparse histogram and sorting it by counts 

    typedef typename thrust::device_vector<T>::const_iterator T_Iterator;

    strided_range<T_Iterator> src(sequence.begin() + sequence_offset, sequence.end(), sequence_itemsize);

    thrust::device_vector<T> data(src.begin(), src.end());  // copy to avoid sorting original

    thrust::sort(data.begin(), data.end());

   // product of sorted data with itself shifted by one finds "edges" between values 
    int num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                             data.begin() + 1,
                                             int(1),
                                             thrust::plus<int>(),
                                             thrust::not_equal_to<T>());

    histogram_values.resize(num_bins);
    histogram_counts.resize(num_bins);
  
    // find the end of each bin of values
    thrust::reduce_by_key(data.begin(), data.end(),
                        thrust::constant_iterator<int>(1),
                        histogram_values.begin(),
                        histogram_counts.begin());
  
    thrust::sort_by_key( histogram_counts.begin(), histogram_counts.end(), histogram_values.begin());
                
}




// hmm seems cannot template this ...
__constant__ unsigned long long dev_lookup[dev_lookup_n]; 


template <typename T>
void update_dev_lookup(T* data) // data needs to have at least dev_lookup_n elements  
{
    cudaMemcpyToSymbol(dev_lookup,data,dev_lookup_n*sizeof(T));
}


template <typename T, typename S>
struct apply_lookup_functor : public thrust::unary_function<T,S>
{
    S m_offset ;
    S m_missing ;

    apply_lookup_functor(S offset, S missing)
        :
        m_offset(offset),    
        m_missing(missing)
        {
        }    


    // host function cannot access __constant__ memory hence device only
    __device__   
    S operator()(T seq)
    {
        S idx(m_missing) ; 
        for(unsigned int i=0 ; i < dev_lookup_n ; i++)
        {
            if(seq == dev_lookup[i]) idx = i + m_offset ;
            // NB not breaking as hope this will keep memory access lined up between threads 
        }
        return idx ; 
    } 
};


template <typename T, typename S>
void apply_histogram_imp(const thrust::device_vector<T>& sequence,
                               unsigned int sequence_offset,
                               unsigned int sequence_itemsize,
                               thrust::device_vector<S>& target,
                               unsigned int target_offset,
                               unsigned int target_itemsize)
{

    typedef typename thrust::device_vector<T>::const_iterator T_Iterator;
    strided_range<T_Iterator> src(sequence.begin() + sequence_offset, sequence.end(), sequence_itemsize);

    typedef typename thrust::device_vector<S>::iterator S_Iterator;
    strided_range<S_Iterator> dest(target.begin() + target_offset, target.end(), target_itemsize);

    S missing = std::numeric_limits<S>::max() ;
    S offset  =  1 ; 

    thrust::transform( src.begin(), src.end(), dest.begin(), apply_lookup_functor<T,S>(offset, missing) ); 
}


template <typename S>
void strided_copyback( unsigned int n, thrust::host_vector<S>& dest, thrust::device_vector<S>& src, unsigned int src_offset, unsigned int src_itemsize )
{ 
    typedef typename thrust::device_vector<S>::iterator Iterator;
    strided_range<Iterator> column(src.begin() + src_offset, src.end(), src_itemsize);
    thrust::copy( column.begin(),  column.begin() + n,   dest.begin() ) ; 
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




// plant symbols via explicit instanciation 

template void strided_copyback<unsigned int>( unsigned int n, 
        thrust::host_vector<unsigned int>& dest, 
        thrust::device_vector<unsigned int>& src, unsigned int src_offset, unsigned int src_itemsize );

template void strided_copyback<unsigned char>( unsigned int n, 
        thrust::host_vector<unsigned char>& dest, 
        thrust::device_vector<unsigned char>& src, unsigned int src_offset, unsigned int src_itemsize );


template void apply_histogram_imp<unsigned long long, unsigned char>(
       const thrust::device_vector<unsigned long long>& sequence,
                                    unsigned int sequence_offset,
                                  unsigned int sequence_itemsize,
             thrust::device_vector<unsigned char>&        target,
                                     unsigned int target_offset,
                                   unsigned int target_itemsize);

template void sparse_histogram_imp<unsigned long long>(
                  const thrust::device_vector<unsigned long long>& history,
                                                     unsigned int history_offset,
                                                     unsigned int history_itemsize,
                        thrust::device_vector<unsigned long long>& histogram_values,
                        thrust::device_vector<int>&                histogram_counts);


template void update_dev_lookup<unsigned long long>(unsigned long long* data);


