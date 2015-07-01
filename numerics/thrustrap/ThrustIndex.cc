#include "ThrustIndex.hh"

#include "stdio.h"
#include <iostream>
#include <iomanip>

#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


template<typename T,typename S>
void ThrustIndex<T,S>::version()
{
    LOG(info) << "ThrustIndex::version with Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION ;
}

template<typename T,typename S>
void ThrustIndex<T,S>::init()
{
    version();
    std::cout << "ThrustIndex::init "
              << " num_elements " <<  m_num_elements  
              << " target_itemsize " << m_target_itemsize  
              << std::endl ;

    unsigned int targetSize = getTargetSize();    
    thrust::device_ptr<S>  tptr = thrust::device_pointer_cast(m_target_devptr) ;
    m_target = thrust::device_vector<S>(tptr, tptr+targetSize);
}

template<typename T,typename S>
void ThrustIndex<T,S>::indexHistory(T* history_devptr, unsigned int target_offset)
{
    m_history = new ThrustHistogram<T,S>(history_devptr, m_num_elements, m_target_itemsize, target_offset) ;
    //m_history->dumpSequence("ThrustIndex::indexHistory", 100);
    m_history->createHistogram();
    m_history->apply( m_target ); 
}

template<typename T,typename S>
void ThrustIndex<T,S>::indexMaterial(T* material_devptr, unsigned int target_offset)
{
    m_material = new ThrustHistogram<T,S>(material_devptr, m_num_elements, m_target_itemsize, target_offset) ;
    //m_material->dumpSequence("ThrustIndex::indexMaterial", 100);
    m_material->createHistogram();
    m_material->apply( m_target ); 
}

template<typename T,typename S>
NPY<S>* ThrustIndex<T,S>::makeTargetArray()
{
    thrust::host_vector<S> target = m_target ;                // full pullback, expensive 
    return NPY<S>::make_scalar(target.size(), target.data()); 
}


template<typename T,typename S>
void ThrustIndex<T,S>::dumpTarget(const char* msg, unsigned int n)
{
    LOG(info) << msg << " " << n ; 

    unsigned int nn = n*m_target_itemsize ; 
    thrust::host_vector<S> target(nn) ;
    thrust::copy( m_target.begin(), m_target.begin() + nn , target.begin());

    unsigned int idx(0) ; 
    for(typename thrust::host_vector<S>::iterator it=target.begin() ; it != target.end() ; it++)
    {
          unsigned int iidx = idx % m_target_itemsize ; 
          if(iidx == 0 ) std::cout << std::setw(5) << idx/m_target_itemsize ; 
          std::cout << std::setw(20) << std::dec << static_cast<unsigned int>(*it) ;
          if(iidx == m_target_itemsize - 1 ) std::cout << std::endl ; 
          idx++ ;
    }
}




// explicit instanciation
template class ThrustIndex<unsigned long long, unsigned char>;

