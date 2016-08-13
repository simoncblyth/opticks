
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

__host__ std::ostream& operator<< (std::ostream& os, const unsigned char& p) 
{
        os << " 0x" << std::hex << int(p) << std::dec << " " ;  
        return os;
}



OBuf::OBuf(const char* name, optix::Buffer& buffer, NPYBase* npy ) : OBufBase(name, buffer, npy)
{
}



template <typename T>
void OBuf::dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end)
{
    Summary(msg);

    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)getDevicePtr()) ; 
    if(m_hexdump) std::cout << std::hex ; 

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
    if(m_hexdump) std::cout << std::dec ; 
}


template <typename T>
void OBuf::dumpint(const char* msg, unsigned int stride, unsigned int begin, unsigned int end)
{

    // dumpint necessitated in addition to dump as streaming unsigned char gives characters not integers

    Summary(msg);

    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)getDevicePtr()) ; 


    thrust::host_vector<T> h ; 

    if( stride == 0 )
    {
        h.resize(thrust::distance(p+begin, p+end)); 
        thrust::copy( p + begin, p + end, h.begin()); 
    }
    else
    {
        typedef typename thrust::device_vector<T>::iterator Iterator;
        strided_range<Iterator> sri( p + begin, p + end, stride );
        h.resize(thrust::distance(sri.begin(), sri.end())); 
        thrust::copy( sri.begin(), sri.end(), h.begin() ); 
    }

    for(unsigned int i=0 ; i < h.size() ; i++)
    {
        std::cout 
                 << std::setw(7) << i 
                 << std::setw(7) << int(h[i])
                 << std::endl ;  
    }
}







template <typename T>
T OBuf::reduce(unsigned int stride, unsigned int begin, unsigned int end )
{
    // hmm this assumes do not do reductions at float4 level ?
    if(end == 0u) end = getNumAtoms(); 

    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)getDevicePtr()) ; 

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


/*
template optix::float4* OBuf::getDevicePtr<optix::float4>();
template optix::uint4* OBuf::getDevicePtr<optix::uint4>();
template unsigned int* OBuf::getDevicePtr<unsigned int>();
template unsigned long long* OBuf::getDevicePtr<unsigned long long>();
*/

template OXRAP_API void OBuf::dump<optix::float4>(const char*, unsigned int, unsigned int, unsigned int);
template OXRAP_API void OBuf::dump<optix::uint4>(const char*, unsigned int, unsigned int, unsigned int);
template OXRAP_API void OBuf::dump<unsigned int>(const char*, unsigned int, unsigned int, unsigned int);
template OXRAP_API void OBuf::dump<unsigned long long>(const char*, unsigned int, unsigned int, unsigned int);
template OXRAP_API void OBuf::dump<unsigned char>(const char*, unsigned int, unsigned int, unsigned int);
template OXRAP_API void OBuf::dump<int>(const char*, unsigned int, unsigned int, unsigned int);

template OXRAP_API void OBuf::dumpint<unsigned char>(const char*, unsigned int, unsigned int, unsigned int);


template OXRAP_API unsigned int OBuf::reduce<unsigned int>(unsigned int, unsigned int, unsigned int);
template OXRAP_API unsigned long long OBuf::reduce<unsigned long long>(unsigned int, unsigned int, unsigned int);



