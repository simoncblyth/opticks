#pragma once

#include <thrust/device_vector.h>
#include "NPY.hpp"
class Index ; 

template <typename T, typename S>
class ThrustHistogram {
    public:
         ThrustHistogram(T* sequence_devptr, unsigned int num_elements, unsigned int target_itemsize, unsigned int target_offset);
    public:
         void createHistogram();
         void apply(thrust::device_vector<S>& target);
    private:
         void pullback(unsigned int n=32);
    public:
         void dumpSequence( const char* msg="ThrustHistogram::dumpSequence", unsigned int n=100);
         void dumpHistogram(const char* msg="ThrustHistogram::dumpHistogram", unsigned int n=100);
    public:
         NPY<T>* makeSequenceArray();
         Index* makeIndex(const char* itemtype);
    private:
         void init();
    private:
         thrust::device_vector<T>   m_sequence ;
    private:
         thrust::device_vector<T>   m_values;
         thrust::device_vector<int> m_counts;
    private:
         thrust::host_vector<T>     m_values_h ;  // copying from device to host 
         thrust::host_vector<int>   m_counts_h ; 
    private:
         T*                         m_sequence_devptr ; 
    private:
         unsigned int               m_num_elements ;
         unsigned int               m_target_itemsize ; 
         unsigned int               m_target_offset ; 

};

template<typename T, typename S>
inline ThrustHistogram<T,S>::ThrustHistogram(T* sequence_devptr, unsigned int num_elements, unsigned int target_itemsize, unsigned int target_offset ) 
    :
    m_sequence_devptr(sequence_devptr),
    m_num_elements(num_elements),
    m_target_itemsize(target_itemsize),
    m_target_offset(target_offset)
{
    init();
}



