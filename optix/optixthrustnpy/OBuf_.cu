
// https://devtalk.nvidia.com/default/topic/574078/?comment=3896854

#include <vector_types.h>

#include "OBuf.hh"

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
OBuf<T>::OBuf(optix::Buffer& buffer ) : m_buffer(buffer), m_device(0u) {}


template <typename T>
unsigned int OBuf<T>::getSize()
{
    RTsize width, height, depth ; 
    m_buffer->getSize(width, height, depth);
    RTsize size = width*height*depth ; 
    return size ; 
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


//template void OBuf<optix::float4>::dump(const char*, unsigned int, unsigned int) ; 

template class OBuf<optix::float4> ;
template class OBuf<optix::uint4> ;
 
