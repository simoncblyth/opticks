#include "TUtil.hh"
#include <thrust/device_vector.h>
#include "CBufSpec.hh"
#include "float4x4.h"



template <typename T>
CBufSpec make_bufspec(const thrust::device_vector<T>& d_vec )
{     
    const T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());

    unsigned int size = d_vec.size() ;
    unsigned int nbytes =  size*sizeof(T) ;
      
    return CBufSpec( (void*)raw_ptr, size, nbytes );
} 



template THRAP_API CBufSpec make_bufspec<unsigned long long>(const thrust::device_vector<unsigned long long>& );
template THRAP_API CBufSpec make_bufspec<unsigned int>(const thrust::device_vector<unsigned int>& );
template THRAP_API CBufSpec make_bufspec<unsigned char>(const thrust::device_vector<unsigned char>& );
template THRAP_API CBufSpec make_bufspec<int>(const thrust::device_vector<int>& );
template THRAP_API CBufSpec make_bufspec<float4>(const thrust::device_vector<float4>& );
template THRAP_API CBufSpec make_bufspec<float>(const thrust::device_vector<float>& );
template THRAP_API CBufSpec make_bufspec<float4x4>(const thrust::device_vector<float4x4>& );



