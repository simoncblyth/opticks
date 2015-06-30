#include "ThrustHistogram.hh"

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
void ThrustHistogram<T>::create()
{
    std::cout << "ThrustHistogram<T>::create num " << m_num  << std::endl;

    thrust::device_ptr<T>  ptr = thrust::device_pointer_cast(m_devptr) ;

    thrust::device_vector<T> input(ptr, ptr+m_num);

    sparse_histogram_imp(input, m_values, m_counts, m_index );
}


template<typename T>
void ThrustHistogram<T>::dump()
{
    thrust::host_vector<T>    values = m_values ; 
    thrust::host_vector<int>  counts = m_counts ; 

    unsigned int total(0) ; 
    for(unsigned int i=0 ; i < values.size() ; i++)
    {
        total += counts[i] ;
        //T seq = values[i];
        //std::string sseq = flags ? flags->getSequenceString(seq) : "" ; 
        if(counts[i] > 1000) std::cout 
                      << std::setw(5) << i 
                      << std::setw(20) << std::hex << values[i]
                      << std::setw(20) << std::dec << counts[i]
                 //   << "  " << sseq
                      << std::endl ; 
    }
    std::cout << "total " << total << std::endl ; 
}


// explicit instanciation
template class ThrustHistogram<unsigned long long>;

