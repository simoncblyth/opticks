#include "ThrustHistogram.hh"
#include "NPY.hpp"
#include "Index.hpp"
#include "stringutil.hpp"

#include <iostream>
#include <iomanip>

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "ThrustHistogramImp.h"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



template<typename T>
void ThrustHistogram<T>::init()
{
    std::cout << "ThrustHistogram<T>::init num " << m_num  << std::endl;

    thrust::device_ptr<T>  ptr = thrust::device_pointer_cast(m_devptr) ;

    m_input = thrust::device_vector<T>(ptr, ptr+m_num);
}


template<typename T>
void ThrustHistogram<T>::create()
{
    sparse_histogram_imp(m_input, m_values, m_counts, m_index );

    m_values_h = m_values ;  // copying back from device to host 
    m_counts_h = m_counts ; 
    // only pullback the large m_input array as needed
}

template<typename T>
void ThrustHistogram<T>::dumpInput(const char* msg, unsigned int n)
{
    thrust::host_vector<T>  input = m_input ;   // copying from device to host
    std::cout << msg << " size " << input.size() ;

    unsigned int nn = n < input.size() ? n : input.size() ; 

    for(unsigned int i=0 ; i < nn ; i++)
    {
          std::cout 
                      << std::setw(5) << i 
                      << std::setw(20) << std::hex << input[i]
                      << std::endl ; 
    }
}


template<typename T>
NPY<T>* ThrustHistogram<T>::makeInputArray()
{
    thrust::host_vector<T>  input = m_input ;   // copying from device to host
    return NPY<T>::make_scalar(input.size(), input.data()); 
}

template<typename T>
Index* ThrustHistogram<T>::makeIndex(const char* itemtype)
{
    Index* index = new Index(itemtype);
    for(unsigned int i=0 ; i < m_values_h.size() ; i++)
    { 
        std::string name = as_hex(m_values_h[i]);
        index->add(name.c_str(), m_counts_h[i] ); 
    }
    return index ; 
}


template<typename T>
void ThrustHistogram<T>::dump()
{
   // dumping hostside histo
    unsigned int total(0) ; 
    for(unsigned int i=0 ; i < m_values_h.size() ; i++)
    {
        total += m_counts_h[i] ;
        if(m_counts_h[i] > 1000) 
                   std::cout 
                      << std::setw(5) << i 
                      << std::setw(20) << std::hex << m_values_h[i]
                      << std::setw(20) << std::dec << m_counts_h[i]
                      << std::endl ; 
    }
    std::cout << "total " << total << std::endl ; 
}


// explicit instanciation
template class ThrustHistogram<unsigned long long>;

