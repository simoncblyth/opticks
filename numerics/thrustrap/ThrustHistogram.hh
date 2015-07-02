#pragma once

#include <thrust/device_vector.h>
#include "NPY.hpp"
#include "string.h"
class Index ; 

template <typename T, typename S>
class ThrustHistogram {
    public:
         ThrustHistogram(const char* itemtype, 
                         unsigned int num_elements, 
                         unsigned int sequence_itemsize, 
                         unsigned int sequence_offset, 
                         unsigned int target_itemsize, 
                         unsigned int targer_offset);
    public:
         void createHistogram(
                    thrust::device_vector<T>& sequence
                   );
         void apply(
                    thrust::device_vector<T>& sequence,
                    thrust::device_vector<S>& target
                  );
         Index* getIndex();
         unsigned int getSequenceSize();
    private:
         void pullback(unsigned int n=32);
    public:
         void dumpHistogram(const char* msg="ThrustHistogram::dumpHistogram", unsigned int n=100);
    public:
         // DEBUG 
         NPY<T>* makeSequenceIndexArray();
    private:
         void init();
         Index* makeIndex(const char* itemtype);
         void   setIndex(Index* index);
    private:
    private:
         thrust::device_vector<T>   m_values;
         thrust::device_vector<int> m_counts;
    private:
         thrust::host_vector<T>     m_values_h ;  // copying from device to host 
         thrust::host_vector<int>   m_counts_h ; 
    private:
         const char*                m_itemtype ;   
         unsigned int               m_num_elements ;
         unsigned int               m_sequence_itemsize ; 
         unsigned int               m_sequence_offset ; 
         unsigned int               m_target_itemsize ; 
         unsigned int               m_target_offset ; 
    private:
         Index*                     m_target_index ; 

};

template<typename T, typename S>
inline ThrustHistogram<T,S>::ThrustHistogram(
         const char* itemtype, 
         unsigned int num_elements, 
         unsigned int sequence_itemsize, 
         unsigned int sequence_offset, 
         unsigned int target_itemsize, 
         unsigned int target_offset ) 
    :
    m_itemtype(strdup(itemtype)),
    m_num_elements(num_elements),
    m_sequence_itemsize(sequence_itemsize),
    m_sequence_offset(sequence_offset),
    m_target_itemsize(target_itemsize),
    m_target_offset(target_offset),
    m_target_index(NULL)
{
    init();
}


template<typename T, typename S>
inline Index* ThrustHistogram<T,S>::getIndex()
{
    return m_target_index  ; 
}


template<typename T, typename S>
inline void ThrustHistogram<T,S>::setIndex(Index* index)
{
    m_target_index = index  ; 
}


template <typename T, typename S>
inline unsigned int ThrustHistogram<T,S>::getSequenceSize()
{
    return m_num_elements*m_sequence_itemsize ; 
}


