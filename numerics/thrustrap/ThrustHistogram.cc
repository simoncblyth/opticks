#include "ThrustHistogram.hh"
#include "NPY.hpp"
#include "Index.hpp"
#include "stringutil.hpp"

#include <iostream>
#include <iomanip>

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reverse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#include <thrust/iterator/reverse_iterator.h>

#include "ThrustHistogramImp.h"
#include "strided_range.h"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

template<typename T,typename S>
void ThrustHistogram<T,S>::init()
{
    std::cout << "ThrustHistogram<<T,S>>::init "
              << " num_elements " << m_num_elements  
              << std::endl ;

    thrust::device_ptr<T>  ptr = thrust::device_pointer_cast(m_sequence_devptr) ;
    m_sequence = thrust::device_vector<T>(ptr, ptr+m_num_elements);
}

template<typename T,typename S>
void ThrustHistogram<T,S>::createHistogram()
{
    sparse_histogram_imp<T>(m_sequence, m_values, m_counts );

    pullback(dev_lookup_n);

    update_dev_lookup<T>( m_values_h.data() );
}

template<typename T,typename S>
void ThrustHistogram<T,S>::pullback(unsigned int n)
{
    // partial pullback, only need small number (~32) of most popular ones on host 
    m_values_h.resize(n);
    m_counts_h.resize(n);

    // what happens when not long enough ?
    thrust::copy( m_values.end() - n, m_values.end(), m_values_h.begin() );
    thrust::copy( m_counts.end() - n, m_counts.end(), m_counts_h.begin() );

    thrust::reverse(m_values_h.begin(), m_values_h.end());
    thrust::reverse(m_counts_h.begin(), m_counts_h.end());
}


template <typename T,typename S>
void ThrustHistogram<T,S>::apply(thrust::device_vector<S>& target)
{
   // apply this histogram lookup to the target at offset/stride specified in ctor
    apply_histogram_imp<T>(m_sequence, target, m_target_offset, m_target_itemsize );
}


template<typename T,typename S>
void ThrustHistogram<T,S>::dumpSequence(const char* msg, unsigned int n)
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



template<typename T,typename S>
NPY<T>* ThrustHistogram<T,S>::makeSequenceArray()
{
    thrust::host_vector<T> history = m_sequence ;   // full pullback, expensive 
    return NPY<T>::make_scalar(history.size(), history.data()); 
}


template<typename T,typename S>
Index* ThrustHistogram<T,S>::makeIndex(const char* itemtype)
{
    // NB host arrays are partial 
    Index* index = new Index(itemtype);
    for(unsigned int i=0 ; i < m_values_h.size() ; i++)
    { 
        std::string name = as_hex(m_values_h[i]);
        index->add(name.c_str(), m_counts_h[i] ); 
    }
    return index ; 
}


template<typename T,typename S>
void ThrustHistogram<T,S>::dumpHistogram(const char* msg, unsigned int n)
{
    LOG(info) << msg ; 

    unsigned int total(0) ; 
    for(unsigned int i=0 ; i < m_values_h.size() ; i++)
    {
        total += m_counts_h[i] ;
        std::cout 
                      << std::setw(5) << i 
                      << std::setw(20) << std::hex << m_values_h[i]
                      << std::setw(20) << std::dec << m_counts_h[i]
                      << std::endl ; 
    }
    std::cout 
            << " histo size " << m_values_h.size() 
            << " total " << total 
            << std::endl ; 
}


// explicit instanciation
template class ThrustHistogram<unsigned long long, unsigned char>;

