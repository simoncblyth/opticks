#pragma once

#include "stdlib.h"
#include "assert.h"

#include "ThrustHistogram.hh"
#include "ThrustArray.hh"

class Index ; 


template <typename T, typename S>
class ThrustIdx {
   public:
         static void version();
         enum { NUM_HIST = 2 };
   public:
         ThrustIdx(ThrustArray<S>* target, ThrustArray<T>* source);
   public:
         void                  makeHistogram(unsigned int offset);
         ThrustHistogram<T,S>* getHistogram(unsigned int offset);
         Index*                getHistogramIndex(unsigned int offset);
   private:
         void init();
   private:
         ThrustArray<S>*           m_target ;
         ThrustArray<T>*           m_source;
   private:
         ThrustHistogram<T,S>*     m_histogram[NUM_HIST] ; 

};

template <typename T, typename S>
inline ThrustIdx<T,S>::ThrustIdx(ThrustArray<S>* target, ThrustArray<T>* source)
    :
    m_target(target),
    m_source(source)
{
    init();
    for(unsigned int i=0 ; i < NUM_HIST ; i++) m_histogram[i] = NULL ; 
}


template <typename T, typename S>
inline ThrustHistogram<T,S>* ThrustIdx<T,S>::getHistogram(unsigned int offset) 
{
    assert(offset < NUM_HIST);
    return m_histogram[offset] ;
}

template <typename T, typename S>
inline Index* ThrustIdx<T,S>::getHistogramIndex(unsigned int offset) 
{
    assert(offset < NUM_HIST);
    return m_histogram[offset]->getIndex() ;
}


