#pragma once

#include <string>

#include "CBufSlice.hh"
#include <thrust/device_vector.h>
class Index ; 

#define TSPARSE_LOOKUP_N 32

#include "THRAP_API_EXPORT.hh"

template <typename T>
class THRAP_API TSparse {
   public:
      TSparse(const char* label, CBufSlice source, bool hexkey=true);
   private:
      void init();
   public:
      void make_lookup(); 
      template <typename S> void apply_lookup(CBufSlice target);
      Index* getIndex();
   private:
      void count_unique();  // creates on device sparse histogram 
      void update_lookup(); // writes small number (eg 32) of most popular uniques to global device constant memory   
      void populate_index(Index* index);
   public:
      std::string dump_(const char* msg="TSparse<T>::dump") const ;
      void dump(const char* msg="TSparse<T>::dump") const ;
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


