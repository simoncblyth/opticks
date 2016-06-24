#include <cstdio>
#include <iterator>
#include <iomanip>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "TBuf.hh"
#include "TUtil.hh"

#include "strided_range.h"
#include "strided_repeated_range.h"

#include "NPY.hpp"



TBuf::TBuf(const char* name, CBufSpec spec ) :
        m_name(strdup(name)),
        m_spec(spec)
{
}

CBufSlice TBuf::slice( unsigned int stride, unsigned int begin, unsigned int end ) const 
{
    if(end == 0u) end = m_spec.size ;  
    return CBufSlice(m_spec.dev_ptr, m_spec.size, m_spec.num_bytes, stride, begin, end);
}

void TBuf::Summary(const char* msg) const 
{
    printf("%s %s \n", msg, m_name );
}

void* TBuf::getDevicePtr() const 
{
    return m_spec.dev_ptr ; 
}
unsigned int TBuf::getNumBytes() const 
{
    return m_spec.num_bytes ; 
}
unsigned int TBuf::getSize() const 
{
    return m_spec.size ; 
}


template <typename T>
void TBuf::download(NPY<T>* npy) const 
{
    unsigned int numBytes = npy->getNumBytes(0) ;

    unsigned int numBytes2 = getNumBytes();

    if(numBytes != numBytes2)
        std::cout << "TBuf::download FATAL numBytes mismatch "
                  << " numBytes " << numBytes 
                  << " numBytes2 " << numBytes2
                  << std::endl ;  

    assert(numBytes == numBytes2);
    void* src = getDevicePtr();
    void* dst = npy->zero();
    cudaMemcpy( dst, src, numBytes, cudaMemcpyDeviceToHost );
}

void TBuf::zero()
{
    cudaMemset( getDevicePtr(), 0, getNumBytes());
}

template <typename T>
void TBuf::dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end ) const 
{
    Summary(msg);

    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)getDevicePtr()) ;

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
void TBuf::dumpint(const char* msg, unsigned int stride, unsigned int begin, unsigned int end) const 
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
T TBuf::reduce(unsigned int stride, unsigned int begin, unsigned int end ) const 
{
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



template <typename T>
void TBuf::repeat_to( TBuf* other, unsigned int stride, unsigned int begin, unsigned int end, unsigned int repeats ) const 
{
    thrust::device_ptr<T> src = thrust::device_pointer_cast((T*)getDevicePtr()) ;
    thrust::device_ptr<T> tgt = thrust::device_pointer_cast((T*)other->getDevicePtr()) ;

    typedef typename thrust::device_vector<T>::iterator Iterator;

    strided_repeated_range<Iterator> si( src + begin, src + end, stride, repeats);

    thrust::copy( si.begin(), si.end(),  tgt );    
}



template void TBuf::dump<int>(const char*, unsigned int, unsigned int, unsigned int) const ;
template void TBuf::dump<unsigned int>(const char*, unsigned int, unsigned int, unsigned int) const ;
template void TBuf::dump<unsigned long long>(const char*, unsigned int, unsigned int, unsigned int) const ;

template void TBuf::dumpint<unsigned char>(const char*, unsigned int, unsigned int, unsigned int) const ;

template void TBuf::repeat_to<unsigned char>(TBuf*, unsigned int, unsigned int, unsigned int, unsigned int) const ;
template unsigned int TBuf::reduce<unsigned int>(unsigned int, unsigned int, unsigned int) const ;

template void TBuf::download<unsigned char>(NPY<unsigned char>*) const ;




