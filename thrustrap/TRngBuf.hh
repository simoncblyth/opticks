#pragma once

/**
TRngBuf
==========

cuRAND GPU generation of random numbers using thrust and NPY 

**/


#include "TBuf.hh"
#include "CBufSpec.hh"
#include "CBufSlice.hh"
#include "plog/Severity.h"

template <typename T> class NPY ; 

#include "THRAP_API_EXPORT.hh" 

template<typename T>
class THRAP_API TRngBuf : public TBuf {

      static const plog::Severity LEVEL ; 
   public:
      TRngBuf(unsigned ni, unsigned nj, CBufSpec spec, unsigned long long seed=0ull, unsigned long long offset=0ull );
      void setIBase(unsigned ibase); 
      unsigned getIBase() const ; 
      void generate();

      __device__ void operator()(unsigned id) ;
   private:
      int  preinit() const ; 
      void init() const ; 
   private:
      void generate(unsigned id_offset, unsigned id_0, unsigned id_1);

   private:
       int      m_preinit ;     
       unsigned m_ibase ;      // base photon index  
       unsigned m_ni ;         // number of photon slots
       unsigned m_nj ;         // number of randoms to precook per photon
       unsigned m_num_elem ;   

       unsigned m_id_offset ; 
       unsigned m_id_max ;     // maximum number of photons to generate the randoms for at once
 
       unsigned long long m_seed ; 
       unsigned long long m_offset ; 
       T*                 m_dev ; 

};



