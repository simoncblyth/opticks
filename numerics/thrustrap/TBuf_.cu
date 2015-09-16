#include "TBuf.hh"

#include "strided_range.h"
#include "strided_repeated_range.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iterator>
#include <iomanip>
#include <iostream>

#include "NPY.hpp"

template <typename T>
void TBuf::download(NPY<T>* npy)
{
    unsigned int numBytes = npy->getNumBytes(0) ;
    assert(numBytes == getNumBytes());
    void* src = getDevicePtr();
    void* dst = npy->zero();
    cudaMemcpy( dst, src, numBytes, cudaMemcpyDeviceToHost );
}


template <typename T>
void TBuf::dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end )
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
void TBuf::dumpint(const char* msg, unsigned int stride, unsigned int begin, unsigned int end)
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
T TBuf::reduce(unsigned int stride, unsigned int begin, unsigned int end )
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
void TBuf::repeat_to( TBuf* other, unsigned int stride, unsigned int begin, unsigned int end, unsigned int repeats )
{
    thrust::device_ptr<T> src = thrust::device_pointer_cast((T*)getDevicePtr()) ;
    thrust::device_ptr<T> tgt = thrust::device_pointer_cast((T*)other->getDevicePtr()) ;

    typedef typename thrust::device_vector<T>::iterator Iterator;

    strided_repeated_range<Iterator> si( src + begin, src + end, stride, repeats);

    thrust::copy( si.begin(), si.end(),  tgt );    
}



template void TBuf::dump<unsigned int>(const char*, unsigned int, unsigned int, unsigned int);
template void TBuf::dumpint<unsigned char>(const char*, unsigned int, unsigned int, unsigned int);
template void TBuf::repeat_to<unsigned char>(TBuf*, unsigned int, unsigned int, unsigned int, unsigned int);
template unsigned int TBuf::reduce<unsigned int>(unsigned int, unsigned int, unsigned int);
template void TBuf::download<unsigned char>(NPY<unsigned char>* npy);



