/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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



