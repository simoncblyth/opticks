#pragma once

#include <thrust/device_vector.h>
#include "NPY.hpp"


template <typename S>
class ThrustArray {
   public:
         ThrustArray(S* devptr, unsigned int num_elements, unsigned int itemsize);
   public:
         unsigned int getSize();
         unsigned int getNumElements();
         unsigned int getItemSize();
   public:
         thrust::device_vector<S>&  getDeviceVector();
   public:
         void dump( const char* msg="ThrustArray::dump", unsigned int nline=100);
   public:
         void copy_to(ThrustArray<S>& other);
         void repeat_to(unsigned int nrepeat, ThrustArray<S>& other);
   public:
         void save(const char* path);
         NPY<S>* makeNPY(); 
   private:
         void init();
   private:
         thrust::device_vector<S>  m_dvec ;
   private:
         S*                        m_devptr ;
         unsigned int              m_num_elements ;
         unsigned int              m_itemsize ;
};

template <typename S>
inline ThrustArray<S>::ThrustArray(S* devptr, unsigned int num_elements, unsigned int itemsize) 
    :
    m_devptr(devptr),
    m_num_elements(num_elements),
    m_itemsize(itemsize)
{
    init();
}


template <typename S>
inline unsigned int ThrustArray<S>::getSize()
{
    return m_num_elements*m_itemsize ; 
}
template <typename S>
inline unsigned int ThrustArray<S>::getNumElements()
{
    return m_num_elements ; 
}
template <typename S>
inline unsigned int ThrustArray<S>::getItemSize()
{
    return m_itemsize ; 
}






template <typename S>
inline thrust::device_vector<S>& ThrustArray<S>::getDeviceVector()
{
    return m_dvec ;
}




