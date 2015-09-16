#pragma once
#include "stdio.h"
#include "CBufSlice.hh"
#include <thrust/device_vector.h>

#define TSPARSE_LOOKUP_N 32 

// (hopefully) a simplification of ThrustIdx, ThrustHistogram

template <typename T>
class TSparse {
   public:
      TSparse(CBufSlice source);
   public:
      void make_lookup(); 
      template <typename S> void apply_lookup(CBufSlice target);
   private:
      void count_unique();  // creates on device sparse histogram 
      void update_lookup();
   public:
      void dump(const char* msg="TSparse<T>::dump");
   private:
      // input buffer slice specification
      CBufSlice   m_source ; 
   private:
      unsigned int                 m_num_unique ; 
      thrust::device_vector<T>     m_values; 
      thrust::device_vector<int>   m_counts; 
   private:
      thrust::host_vector<T>       m_values_h ;  
      thrust::host_vector<int>     m_counts_h ; 

};

template <typename T>
inline TSparse<T>::TSparse(CBufSlice source ) :
        m_source(source),
        m_num_unique(0u)
{
}





