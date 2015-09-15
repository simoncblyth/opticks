#include "TBuf.hh"

#include "strided_range.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iterator>
#include <iomanip>
#include <iostream>









template <typename T>
void TBuf::dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end )
{
    Summary(msg);

    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)m_spec.dev_ptr) ;

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
T TBuf::reduce(unsigned int stride, unsigned int begin, unsigned int end )
{
    thrust::device_ptr<T> p = thrust::device_pointer_cast((T*)m_spec.dev_ptr) ;

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








template void TBuf::dump<unsigned int>(const char*, unsigned int, unsigned int, unsigned int);
template unsigned int TBuf::reduce<unsigned int>(unsigned int, unsigned int, unsigned int);



