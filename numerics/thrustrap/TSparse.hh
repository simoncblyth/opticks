#pragma once

#include "stdio.h"
#include "CBufSlice.hh"
#include <thrust/device_vector.h>
class Index ; 

#define TSPARSE_LOOKUP_N 32 

// (hopefully) a simplification of ThrustIdx, ThrustHistogram

template <typename T>
class TSparse {
   public:
      TSparse(const char* label, CBufSlice source);
   public:
      void make_lookup(); 
      template <typename S> void apply_lookup(CBufSlice target);
      Index* getIndex();
      void setHexDump(bool hexdump=true);
   private:
      void count_unique();  // creates on device sparse histogram 
      void update_lookup(); // writes small number (eg 32) of most popular uniques to global device constant memory   
      Index* make_index();
   public:
      void dump(const char* msg="TSparse<T>::dump");
   private:
      // input buffer slice specification
      const char* m_label ; 
      CBufSlice   m_source ; 
   private:
      unsigned int                 m_num_unique ; 
      thrust::device_vector<T>     m_values; 
      thrust::device_vector<int>   m_counts; 
   private:
      thrust::host_vector<T>       m_values_h ;  
      thrust::host_vector<int>     m_counts_h ; 
      Index*                       m_index_h ; 
      bool                         m_hexdump ; 

};

template <typename T>
inline TSparse<T>::TSparse(const char* label, CBufSlice source ) :
        m_label(strdup(label)),
        m_source(source),
        m_num_unique(0u),
        m_index_h(NULL),
        m_hexdump(true)
{
}

template <typename T>
inline Index* TSparse<T>::getIndex()
{
    return m_index_h ;
}

template <typename T>
inline void TSparse<T>::setHexDump(bool hexdump)
{
    m_hexdump = hexdump  ;
}










