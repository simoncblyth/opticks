#pragma once

#include <thrust/device_vector.h>
#include "NPY.hpp"


template <typename S>
class ThrustArray {
   public:
         ThrustArray(S* devptr, unsigned int num_elements, unsigned int itemsize); // preexisting buffer on device
   public:
         unsigned int getSize();        //  product of NumElements and ItemSize
         unsigned int getItemSize();    //  typically a small number eg 2 for sequence buffer of unsigned long long (like NPY shape1*shape2) 
         unsigned int getNumElements(); //  typically a large number (like NPY shape0) 
         unsigned int getNumBytes();    //  sizeof(S)*getSize()
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
         void download(NPY<S>* npy);
   private:
         void init();
   private:
         thrust::device_vector<S>  m_dvec ;   // **COPY : DOES NOT SHARE DEVICE STORAGE WITH ORIGINAL DEVPTR**
   private:
         S*                        m_devptr ;
         NPY<S>*                   m_npy ;
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
inline unsigned int ThrustArray<S>::getNumBytes()
{
    return m_num_elements*m_itemsize*sizeof(S) ; 
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




