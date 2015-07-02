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


}

template<typename T,typename S>
void ThrustHistogram<T,S>::createHistogram( thrust::device_vector<T>& sequence)
{
    sparse_histogram_imp<T>(sequence, 
                            m_sequence_offset, 
                            m_sequence_itemsize, 
                            m_values, 
                            m_counts );

    pullback(dev_lookup_n);

    update_dev_lookup<T>( m_values_h.data() );
}


template <typename T,typename S>
void ThrustHistogram<T,S>::apply(
          thrust::device_vector<T>& sequence, 
          thrust::device_vector<S>& target
   )
{
   // apply this histogram lookup to the target at offset/stride specified in ctor
    apply_histogram_imp<T>(sequence, 
                           m_sequence_offset,
                           m_sequence_itemsize,
                           target, 
                           m_target_offset, 
                           m_target_itemsize );

    Index* index = makeIndex(m_itemtype);
    setIndex(index);
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

template<typename T,typename S>
NPY<T>* ThrustHistogram<T,S>::makeSequenceIndexArray()
{
    unsigned int size = m_values.size() ;
    pullback(size);  
    // full pullback   : THIS IS ONLY NEEDED FOR DEBUGGING

    unsigned int ni = size ;
    unsigned int nj = 1 ; 
    unsigned int nk = 2 ; 

    std::vector<T> values ;  
    for(unsigned int i=0 ; i < ni ; i++)
    {
         values.push_back(m_values_h[i]);
         values.push_back(m_counts_h[i]);
    }
    return NPY<T>::make(ni, nj, nk, values.data() ); 
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

