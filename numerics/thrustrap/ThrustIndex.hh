#pragma once

#include "stdlib.h"
#include "ThrustHistogram.hh"
class Index ; 


template <typename T, typename S>
class ThrustIndex {
   public:
         static void version();
   public:
         ThrustIndex(T* sequence_devptr, S* target_devptr, unsigned int num_elements, unsigned int sequence_itemsize=2, unsigned int target_itemsize=4);
   public:
         unsigned int getSequenceSize();
         unsigned int getTargetSize();
   public:
         void indexHistory( unsigned int offset=0 );
         void indexMaterial(unsigned int offset=1 );
         void dumpTarget(   const char* msg="ThrustIndex::dumpTarget", unsigned int n=20);
         void dumpSequence( const char* msg="ThrustIndex::dumpSequence", unsigned int n=100);
   public:
         ThrustHistogram<T,S>* getHistory();
         ThrustHistogram<T,S>* getMaterial();
         Index*                getHistoryIndex();
         Index*                getMaterialIndex();
   public:
         NPY<T>*               makeSequenceArray();
         NPY<S>*               makeTargetArray();
   private:
         void init();
   private:
         thrust::device_vector<T>  m_sequence ;
         thrust::device_vector<S>  m_target;
   private:
         T*                        m_sequence_devptr ;
         S*                        m_target_devptr ;
         unsigned int              m_num_elements ;
   private:
         unsigned int              m_sequence_itemsize ;
         unsigned int              m_target_itemsize ;
   private:
         ThrustHistogram<T,S>*     m_history ; 
         ThrustHistogram<T,S>*     m_material ; 


};

template <typename T, typename S>
inline ThrustIndex<T,S>::ThrustIndex(T* sequence_devptr, S* target_devptr, unsigned int num_elements, unsigned int sequence_itemsize, unsigned int target_itemsize) 
    :
    m_sequence_devptr(sequence_devptr),
    m_target_devptr(target_devptr),
    m_num_elements(num_elements),
    m_sequence_itemsize(sequence_itemsize),
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
inline unsigned int ThrustIndex<T,S>::getSequenceSize()
{
    return m_num_elements*m_sequence_itemsize ; 
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



template <typename T, typename S>
inline Index* ThrustIndex<T,S>::getMaterialIndex() 
{
    return m_material->getIndex() ;
}
template <typename T, typename S>
inline Index* ThrustIndex<T,S>::getHistoryIndex() 
{
    return m_history->getIndex() ;
}

