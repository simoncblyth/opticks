#include "ThrustArray.hh"

#include "stdio.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#include "ThrustArrayImp.h"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

template<typename S>
void ThrustArray<S>::init()
{
    

    unsigned int size = getSize();    
    std::cout << "ThrustArray::init "
              << " num_elements " <<  m_num_elements  
              << " itemsize " <<  m_itemsize  
              << " size " << size  
              << std::endl ;

    assert(m_devptr && "devptr is NULL");
    thrust::device_ptr<S>  ptr = thrust::device_pointer_cast(m_devptr) ;
    m_dvec = thrust::device_vector<S>(ptr, ptr+size);
}

template<typename S>
NPY<S>* ThrustArray<S>::makeNPY()
{
    thrust::host_vector<S> hvec = m_dvec ;   // full pullback, expensive 
    return NPY<S>::make_scalar(hvec.size(), hvec.data()); 
}

template<typename S>
void ThrustArray<S>::dump(const char* msg, unsigned int nitems)
{
    unsigned long req = nitems*m_itemsize ;
    unsigned long nn = std::min(req, m_dvec.size()) ; 
    LOG(info) << msg 
              << " nitems " << nitems  
              << " req " << req  
              << " nn " << nn 
              ; 
    thrust::host_vector<S> hvec(nn) ;
    thrust::copy( m_dvec.begin(), m_dvec.begin() + nn , hvec.begin());

    unsigned int idx(0) ; 
    for(typename thrust::host_vector<S>::iterator it=hvec.begin() ; it != hvec.end() ; it++)
    {
          unsigned int iidx = idx % m_itemsize ; 
          if(iidx == 0 ) std::cout << std::setw(5) << idx/m_itemsize ; 
          std::cout << std::setw(20) << std::dec << static_cast<unsigned int>(*it) ;
          if(iidx == m_itemsize - 1 ) std::cout << std::endl ; 
          idx++ ;
    }
}


template<typename S>
void ThrustArray<S>::copy_to(ThrustArray<S>& other)
{
    thrust::copy( m_dvec.begin(), m_dvec.end(),  other.getDeviceVector().begin() );    
}


template<typename S>
void ThrustArray<S>::repeat_to(unsigned int repeat, ThrustArray<S>& other)
{
    thrust::device_vector<S>& ovec = other.getDeviceVector();
    repeat_to_imp( m_itemsize, repeat, m_dvec, ovec );
}



// explicit instanciation
template class ThrustArray<unsigned char>;


