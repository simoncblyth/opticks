#pragma once

#include <thrust/device_vector.h>
//
// TODO: 
//      arrange for histogram to live in __constant__ memory  and use from 
//      lookup functor 
//
// having templates of templates is painful, eg 
// passing templated class instances to a template method ...
// ... so try to avoid passing things around
// use simple parametrs and transmogify into the templated forms
// where needed ?  
//
// but the trouble is the split between .cu nvcc and .cc clang 
// inevitably means have to pass things  
//
// currently just going fully specialized to avoid one level 
// of templates
//


#include "NPY.hpp"
class Index ; 

template <typename T, typename S>
class ThrustHistogram {
    public:
         ThrustHistogram(T* history_devptr, S* target_devptr, unsigned int num_elements);
    public:
         void createHistogram();
         void apply();
         void pullback(unsigned int n=32);
    public:
         void dumpHistogram(const char* msg="ThrustHistogram::dumpHistogram", unsigned int cutoff=1000);
         void dumpHistory(  const char* msg="ThrustHistogram::dumpHistory", unsigned int n=100);
         void dumpTarget(   const char* msg="ThrustHistogram::dumpTarget", unsigned int n=100);
         void dumpHistoryTarget(const char* msg="ThrustHistogram::dumpHistoryTarget", unsigned int n=100);
    public:
         NPY<T>* makeHistoryArray();
         Index* makeIndex(const char* itemtype);
    private:
         void init();
    private:
         thrust::device_vector<T>   m_history ;
         thrust::device_vector<S>   m_target;
    private:
         thrust::device_vector<T>   m_values;
         thrust::device_vector<int> m_counts;
         thrust::device_vector<int> m_index;
    private:
         thrust::host_vector<T>     m_values_h ;  // copying from device to host 
         thrust::host_vector<int>   m_counts_h ; 
    private:
         T*                         m_history_devptr ; 
         S*                         m_target_devptr ;
         unsigned int               m_num_elements ;
    private:

};

template<typename T, typename S>
inline ThrustHistogram<T,S>::ThrustHistogram(T* history_devptr, S* target_devptr, unsigned int num_elements ) 
    :
    m_history_devptr(history_devptr),
    m_target_devptr(target_devptr),
    m_num_elements(num_elements)
{
    init();
}


