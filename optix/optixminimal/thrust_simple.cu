// nvcc thrust_simple.cu -c -o /tmp/util.o
// interestingly do not need to tell nvcc where to find the headers

#include "thrust_simple.hh"
#include <thrust/device_vector.h>
#include <thrust/count.h>

template <typename T>
T thrust_simple_count( T* d_ptr, unsigned int size, T value )
{
    thrust::device_ptr<T> dptr = thrust::device_pointer_cast(d_ptr);
    thrust::device_vector<T> dvec(dptr, dptr+size);
    T num = thrust::count(dvec.begin(), dvec.end(), value );
    return num ; 
}


// explicit instantiation
template unsigned char thrust_simple_count<unsigned char>(unsigned char*, unsigned int, unsigned char);
