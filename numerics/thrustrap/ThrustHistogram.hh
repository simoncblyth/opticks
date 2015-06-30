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


template <typename T>
class ThrustHistogram {
    public:
         ThrustHistogram(T* devptr, unsigned int num_elements);
    public:
         void create();
         void dump();
    private:
         thrust::device_vector<T>   m_values;
         thrust::device_vector<int> m_counts;
         thrust::device_vector<int> m_index;
    private:
         T*                         m_devptr ; 
         unsigned int               m_num ;

};

template<typename T>
inline ThrustHistogram<T>::ThrustHistogram(T* devptr, unsigned int num_elements ) 
    :
    m_devptr(devptr),
    m_num(num_elements)
{
}


