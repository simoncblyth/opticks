#pragma once

#include "stdio.h"
#include "CBufSlice.hh"
#include <thrust/device_vector.h>
class Index ; 

#define TSPARSE_LOOKUP_N 32
template <typename T>
class TSparse {
   public:
      TSparse(const char* label, CBufSlice source, bool hexkey=true);
   public:
      void make_lookup(); 
      template <typename S> void apply_lookup(CBufSlice target);
      Index* getIndex();
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
      bool                         m_hexkey ; 

};

template <typename T>
inline TSparse<T>::TSparse(const char* label, CBufSlice source, bool hexkey ) :
        m_label(strdup(label)),
        m_source(source),
        m_num_unique(0u),
        m_index_h(NULL),
        m_hexkey(hexkey)
{
}

template <typename T>
inline Index* TSparse<T>::getIndex()
{
    return m_index_h ;
}


