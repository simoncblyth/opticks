#include "ThrustIdx.hh"

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
void ThrustIdx<T,S>::version()
{
    LOG(debug) << "ThrustIdx::version with Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION ;
}

template<typename T,typename S>
void ThrustIdx<T,S>::init()
{
    version();
}

template<typename T,typename S>
void ThrustIdx<T,S>::makeHistogram(unsigned int offset, const char* indextype)
{
    assert(offset < NUM_HIST);

    unsigned int num_elements = m_source->getNumElements();
    assert(num_elements == m_target->getNumElements()); 

    unsigned int source_itemsize = m_source->getItemSize();
    unsigned int target_itemsize = m_target->getItemSize();

    thrust::device_vector<T>& source = m_source->getDeviceVector(); 
    thrust::device_vector<S>& target = m_target->getDeviceVector(); 

    m_histogram[offset] = new ThrustHistogram<T,S>(indextype, num_elements, source_itemsize, offset, target_itemsize, offset) ;
    m_histogram[offset]->createHistogram( source  );
    m_histogram[offset]->apply( source, target ); 
}



// explicit instanciation
template class ThrustIdx<unsigned long long, unsigned char>;

