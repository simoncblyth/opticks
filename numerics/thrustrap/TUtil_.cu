#include "TUtil.hh"
#include <thrust/device_vector.h>
#include "CBufSpec.hh"


template <typename T>
CBufSpec make_bufspec(const thrust::device_vector<T>& d_vec )
{     
    const T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());

    unsigned int size = d_vec.size() ;
    unsigned int nbytes =  size*sizeof(T) ;
      
    return CBufSpec( (void*)raw_ptr, size, nbytes );
} 



template CBufSpec make_bufspec<unsigned long long>(const thrust::device_vector<unsigned long long>& );
template CBufSpec make_bufspec<unsigned int>(const thrust::device_vector<unsigned int>& );
template CBufSpec make_bufspec<unsigned char>(const thrust::device_vector<unsigned char>& );
template CBufSpec make_bufspec<int>(const thrust::device_vector<int>& );
template CBufSpec make_bufspec<float4>(const thrust::device_vector<float4>& );



