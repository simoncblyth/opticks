#pragma once

/**
TRngBuf
==========

cuRAND GPU generation of random numbers using thrust and NPY 

**/


#include "TBuf.hh"
#include "CBufSpec.hh"
#include "CBufSlice.hh"

template <typename T> class NPY ; 

#include "THRAP_API_EXPORT.hh" 

template<typename T>
class THRAP_API TRngBuf : public TBuf {
   public:
      TRngBuf(unsigned ni, unsigned nj, CBufSpec spec, unsigned long long seed=0ull, unsigned long long offset=0ull );
      void generate();

      __device__ void operator()(unsigned id) ;
   private:
      void generate(unsigned id_offset, unsigned id_0, unsigned id_1);

   private:
       unsigned m_ni ; 
       unsigned m_nj ; 

       unsigned m_num_elem ; 
       unsigned m_id_offset ; 
       unsigned m_id_max ; 
 
       unsigned long long m_seed ; 
       unsigned long long m_offset ; 
       T*                 m_dev ; 

};




