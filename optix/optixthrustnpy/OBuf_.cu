
// https://devtalk.nvidia.com/default/topic/574078/?comment=3896854

#include <vector_types.h>

#include "OBuf.hh"

#include "strided_range.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iterator>
#include <iomanip>
#include <iostream>

__host__ std::ostream& operator<< (std::ostream& os, const optix::float4& p) 
{
        os << "[ " 
           << std::setw(10) << p.x << " " 
           << std::setw(10) << p.y << " "
           << std::setw(10) << p.z << " "
           << std::setw(10) << p.w << " "
           << " ]";
         return os;
}

__host__ std::ostream& operator<< (std::ostream& os, const optix::uint4& p) 
{
        os << "[ " 
           << std::setw(10) << p.x << " " 
           << std::setw(10) << p.y << " "
           << std::setw(10) << p.z << " "
           << std::setw(10) << p.w << " "
           << " ]";
         return os;
}



template <typename T>
T* OBuf<T>::getDevicePtr()
{
    CUdeviceptr cu_ptr = m_buffer->getDevicePointer(m_device) ;

    return (T*)cu_ptr ; 
}

template <typename T>
void OBuf<T>::dump(const char* msg, unsigned int begin, unsigned int end )
{
    thrust::device_ptr<T> p = thrust::device_pointer_cast(getDevicePtr()) ; 
    thrust::copy( p + begin, p + end, std::ostream_iterator<T>(std::cout, " \n") ); 
}

template <typename T>
void OBuf<T>::dump_strided(const char* msg, unsigned int begin, unsigned int end, unsigned int stride)
{
    thrust::device_ptr<T> p = thrust::device_pointer_cast(getDevicePtr()) ; 

    typedef typename thrust::device_vector<T>::iterator Iterator;

    strided_range<Iterator> sri( p + begin, p + end, stride );

    thrust::copy( sri.begin(), sri.end(), std::ostream_iterator<T>(std::cout, " \n") ); 
}



template class OBuf<optix::float4> ;
template class OBuf<optix::uint4> ;
template class OBuf<unsigned int> ;
 
