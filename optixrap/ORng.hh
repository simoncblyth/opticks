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

class Opticks ; 
class OContext ; 

#include <vector>
#include "OXPPNS.hh"
#include "plog/Severity.h"
class cuRANDWrapper ; 

#include "OXRAP_API_EXPORT.hh"

/**
ORng
====

Uploads persisted curand rng_states to GPU.
Canonical instance m_orng is ctor resident of OPropagator.

Work is mainly done by cudarap-/cuRANDWrapper

TODO: investigate Thrust based alternatives for curand initialization 
      potential for eliminating cudawrap- 

**/


class OXRAP_API ORng 
{
   public:
      static const plog::Severity LEVEL ; 
   public:
      ORng(Opticks* ok, OContext* ocontext);
   private:
      void init(); 
   private:
      Opticks*        m_ok ; 
      const std::vector<unsigned>& m_mask ; 
      OContext*       m_ocontext ; 
      optix::Context  m_context ;
    protected:
      optix::Buffer   m_rng_states ;
      cuRANDWrapper*  m_rng_wrapper ;


};
