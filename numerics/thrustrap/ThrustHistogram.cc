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

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



template<typename T, typename S>
void ThrustHistogram<T,S>::init()
{
    std::cout << "ThrustHistogram<T>::init num " << m_num_elements  << std::endl;

    thrust::device_ptr<T>  hptr = thrust::device_pointer_cast(m_history_devptr) ;
    m_history = thrust::device_vector<T>(hptr, hptr+m_num_elements);

    thrust::device_ptr<S>  tptr = thrust::device_pointer_cast(m_target_devptr) ;
    m_target = thrust::device_vector<S>(tptr, tptr+m_num_elements);
}



template<typename T, typename S>
void ThrustHistogram<T,S>::createHistogram()
{
    sparse_histogram_imp(m_history, m_values, m_counts, m_index );

    pullback(dev_lookup_n);

    update_dev_lookup( m_values_h.data() );
}

template<typename T, typename S>
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

    /*
    CANNOT GET REVERSE ITERATOR APPROACH TO COMPILE

    // http://stackoverflow.com/questions/26730967/get-reverse-iterator-from-device-ptr-in-cuda

    typedef typename thrust::device_vector<T>::iterator ItV;
    thrust::reverse_iterator<ItV> itv = thrust::make_reverse_iterator(m_values.end());
    thrust::copy( itv, itv+n, m_values_h.begin());

    typedef typename thrust::device_vector<int>::iterator ItC;
    thrust::reverse_iterator<ItC> itc = thrust::make_reverse_iterator(m_counts.end());
    thrust::copy( itc, itc+n, m_counts_h.begin());
    */

}


template<typename T, typename S>
void ThrustHistogram<T,S>::apply()
{
    apply_histogram_imp(m_history, m_values, m_counts, m_index, m_target );
}

template<typename T,typename S>
void ThrustHistogram<T,S>::dumpHistoryTarget(const char* msg, unsigned int n)
{
    LOG(info) << msg << " " << n ; 

    thrust::host_vector<T> history(n) ;
    thrust::host_vector<S> target(n) ;
    thrust::copy( m_history.begin(), m_history.begin() + n , history.begin());
    thrust::copy( m_target.begin(),  m_target.begin() + n,   target.begin()) ; 

    for(unsigned int i=0 ; i < n ; i++ )
    {
          std::cout 
                      << std::setw(5) << i
                      << std::setw(20) << std::hex << history[i]
                      << std::setw(20) << std::dec << target[i]
                      << std::endl ; 
    }
}



template<typename T,typename S>
void ThrustHistogram<T,S>::dumpHistory(const char* msg, unsigned int n)
{
    LOG(info) << msg << " " << n ; 

    thrust::host_vector<T> history(n) ;
    thrust::copy( m_history.begin(), m_history.begin() + n , history.begin());

    unsigned int idx(0) ; 
    for(typename thrust::host_vector<T>::iterator it=history.begin() ; it != history.end() ; it++)
    {
          std::cout 
                      << std::setw(5) << idx 
                      << std::setw(20) << std::hex << *it
                      << std::endl ; 
    }
}


template<typename T,typename S>
void ThrustHistogram<T,S>::dumpTarget(const char* msg, unsigned int n)
{
    LOG(info) << msg << " " << n ; 

    thrust::host_vector<S> target(n) ;
    thrust::copy(m_target.begin(), m_target.begin() + n, target.begin()) ; // hmm what if m_target not that long ?

    unsigned int idx(0) ; 
    for(typename thrust::host_vector<S>::iterator it=target.begin() ; it != target.end() ; it++)
    {
          S value = *it ;   
          std::cout 
                      << std::setw(5) << idx 
                      << std::setw(20) << std::dec << value
                      << std::setw(20) << std::hex << value
                      << std::endl ; 
          idx++ ;
    }
}


template<typename T,typename S>
NPY<T>* ThrustHistogram<T,S>::makeHistoryArray()
{
    thrust::host_vector<T> history = m_history ;   // full pullback, expensive 
    return NPY<T>::make_scalar(history.size(), history.data()); 
}

template<typename T,typename S>
Index* ThrustHistogram<T,S>::makeIndex(const char* itemtype)
{
    // NB the host _h arrays are partial 

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
template class ThrustHistogram<unsigned long long, unsigned int>;

