#include "OBufPair.hh"
#include "OBuf.hh"
#include "strided_range.h"
#include "iexpand.h"



template <typename T>
void OBufPair<T>::seedDestination()
{
    typedef typename thrust::device_vector<T>::iterator Iterator;

    thrust::device_ptr<T> psrc = thrust::device_pointer_cast(m_src.buf->getDevicePtr<T>()) ; 
    thrust::device_ptr<T> pdst = thrust::device_pointer_cast(m_dst.buf->getDevicePtr<T>()) ; 

    strided_range<Iterator> si( psrc + m_src.begin, psrc + m_src.end, m_src.stride );
    strided_range<Iterator> di( pdst + m_dst.begin, pdst + m_dst.end, m_dst.stride );

    iexpand( si.begin(), si.end(), di.begin(), di.end() );

#ifdef DEBUG
    std::cout << "OBufPair<T>::seedDestination " << std::endl ; 
    thrust::copy( di.begin(), di.end(), std::ostream_iterator<T>(std::cout, " ") ); 
    std::cout << "OBufPair<T>::seedDestination " << std::endl ; 
#endif

}


template class OBufPair<unsigned int> ;

