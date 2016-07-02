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
              << std::endl ;

    unsigned int targetSize = getTargetSize();    
    std::cout << "ThrustIndex::init "
              << " target_itemsize " << m_target_itemsize  
              << " targetSize " << targetSize  
              << std::endl ;

    thrust::device_ptr<S>  tptr = thrust::device_pointer_cast(m_target_devptr) ;
    m_target = thrust::device_vector<S>(tptr, tptr+targetSize);

    unsigned int sequenceSize = getSequenceSize();    
    std::cout << "ThrustIndex::init "
              << " sequence_itemsize " << m_sequence_itemsize  
              << " sequenceSize " << sequenceSize  
              << std::endl ;

    thrust::device_ptr<T>  sptr = thrust::device_pointer_cast(m_sequence_devptr) ;
    m_sequence = thrust::device_vector<T>(sptr, sptr+sequenceSize);

}

template<typename T,typename S>
void ThrustIndex<T,S>::indexHistory(unsigned int offset)
{
    m_history = new ThrustHistogram<T,S>("ThrustHistogramHistory", m_num_elements, m_sequence_itemsize, offset, m_target_itemsize, offset) ;
    m_history->createHistogram( m_sequence );
    m_history->apply( m_sequence, m_target ); 
}

template<typename T,typename S>
void ThrustIndex<T,S>::indexMaterial(unsigned int offset)
{
    m_material = new ThrustHistogram<T,S>("ThrustHistogramMaterial", m_num_elements, m_sequence_itemsize, offset, m_target_itemsize, offset) ;
    m_material->createHistogram( m_sequence );
    m_material->apply( m_sequence, m_target ); 
}

template<typename T,typename S>
NPY<S>* ThrustIndex<T,S>::makeTargetArray()
{
    thrust::host_vector<S> target = m_target ;                // full pullback, expensive 
    return NPY<S>::make(target.size(),1,1, target.data()); 
}


template<typename T,typename S>
NPY<T>* ThrustIndex<T,S>::makeSequenceArray()
{
    thrust::host_vector<T> history = m_sequence ;   // full pullback, expensive 
    // ONLY FOR DEBUGGING
    return NPY<T>::make(history.size(),1,1, history.data()); 
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


template<typename T,typename S>
void ThrustIndex<T,S>::dumpSequence(const char* msg, unsigned int n)
{
    LOG(info) << msg << " " << n ; 

    thrust::host_vector<T> sequence(n) ;
    thrust::copy( m_sequence.begin(), m_sequence.begin() + n , sequence.begin());

    unsigned int idx(0) ; 
    for(typename thrust::host_vector<T>::iterator it=sequence.begin() ; it != sequence.end() ; it++)
    {
          std::cout 
                      << std::setw(5) << idx 
                      << std::setw(20) << std::hex << *it
                      << std::endl ; 
    }
}





// explicit instanciation
template class ThrustIndex<unsigned long long, unsigned char>;

