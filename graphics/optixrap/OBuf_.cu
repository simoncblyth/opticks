
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
T* OBuf::getDevicePtr()
{
    CUdeviceptr cu_ptr = m_buffer->getDevicePointer(m_device) ;

    return (T*)cu_ptr ; 
}

template <typename T>
void OBuf::dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end )
{
    Summary(msg);

    thrust::device_ptr<T> p = thrust::device_pointer_cast(getDevicePtr<T>()) ; 

    if( stride == 0 )
    {
        thrust::copy( p + begin, p + end, std::ostream_iterator<T>(std::cout, " \n") ); 
    }
    else
    {
        typedef typename thrust::device_vector<T>::iterator Iterator;
        strided_range<Iterator> sri( p + begin, p + end, stride );
        thrust::copy( sri.begin(), sri.end(), std::ostream_iterator<T>(std::cout, " \n") ); 
    }
}



template <typename T>
T OBuf::reduce(unsigned int stride, unsigned int begin, unsigned int end )
{
    // hmm this assumes do not do reductions at float4 level ?
    if(end == 0u) end = getNumAtoms(); 

    thrust::device_ptr<T> p = thrust::device_pointer_cast(getDevicePtr<T>()) ; 

    T result ; 
    if( stride == 0 )
    {
        result = thrust::reduce( p + begin, p + end ); 
    }
    else
    {
        typedef typename thrust::device_vector<T>::iterator Iterator;
        strided_range<Iterator> sri( p + begin, p + end, stride );
        result = thrust::reduce( sri.begin(), sri.end() ); 
    }
    return result ; 
}






//
// Using a templated class rather than templated member functions 
// has the advantage of only having to explicitly instanciate the class::
//
//    template class OBuf<optix::float4> ;
//    template class OBuf<optix::uint4> ;
//    template class OBuf<unsigned int> ;
//
// as opposed to having to explicly instanciate all the member functions.
//
// But when want differently typed "views" of the 
// same data it seems more logical to used templated member functions.
//

template optix::float4* OBuf::getDevicePtr<optix::float4>();
template optix::uint4* OBuf::getDevicePtr<optix::uint4>();
template unsigned int* OBuf::getDevicePtr<unsigned int>();

template void OBuf::dump<optix::float4>(const char*, unsigned int, unsigned int, unsigned int);
template void OBuf::dump<optix::uint4>(const char*, unsigned int, unsigned int, unsigned int);
template void OBuf::dump<unsigned int>(const char*, unsigned int, unsigned int, unsigned int);


template unsigned int OBuf::reduce<unsigned int>(unsigned int, unsigned int, unsigned int);



