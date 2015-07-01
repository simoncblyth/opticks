#pragma once

#include "stdlib.h"
#include "ThrustHistogram.hh"


template <typename T, typename S>
class ThrustIndex {
   public:
       static void version();
   public:
       ThrustIndex(S* target_devptr, unsigned int num_elements, unsigned int target_itemsize=4);
       unsigned int getTargetSize();
   public:
       void indexHistory( T* history_devptr,  unsigned int target_offset=0 );
       void indexMaterial(T* material_devptr, unsigned int target_offset=1 );
       void dumpTarget(   const char* msg="ThrustIndex::dumpTarget", unsigned int n=20);
   public:
       ThrustHistogram<T,S>* getHistory();
       ThrustHistogram<T,S>* getMaterial();
   public:
       NPY<S>* makeTargetArray();

   private:
       ThrustHistogram<T,S>* m_history ; 
       ThrustHistogram<T,S>* m_material ; 
   private:
         void init();
   private:
         thrust::device_vector<S> m_target;
   private:
         S*                       m_target_devptr ;
         unsigned int             m_num_elements ;
         unsigned int             m_target_itemsize ;
};

template <typename T, typename S>
inline ThrustIndex<T,S>::ThrustIndex(S* target_devptr, unsigned int num_elements, unsigned int target_itemsize) 
    :
    m_target_devptr(target_devptr),
    m_num_elements(num_elements),
    m_target_itemsize(target_itemsize),
    m_history(NULL),
    m_material(NULL)
{
    init();
}


template <typename T, typename S>
inline unsigned int ThrustIndex<T,S>::getTargetSize()
{
    return m_num_elements*m_target_itemsize ; 
}
template <typename T, typename S>
inline ThrustHistogram<T,S>* ThrustIndex<T,S>::getHistory() 
{
    return m_history ;
}
template <typename T, typename S>
inline ThrustHistogram<T,S>* ThrustIndex<T,S>::getMaterial() 
{
    return m_material ;
}

