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
OpSeeder
=========

Distributes unsigned int genstep indices 0:m_num_gensteps-1 into the first 
4 bytes of the 4*float4 photon record in the photon buffer 
using the number of photons per genstep obtained from the genstep buffer 

Note that this is done almost entirely on the GPU, only the num_photons reduction
needs to come back to CPU in order to allocate an appropriately sized OptiX photon 
buffer on GPU.

This per-photon genstep index is used by OptiX photon propagation 
program cu/generate.cu to access the appropriate values from the genstep buffer

When operating with OpenGL buffers the buffer_id lodged in NPYBase is 
all thats needed to reference the GPU buffers.  

When operating without OpenGL need some equivalent way to hold onto the 
GPU buffers or somehow pass them to OptiX 
The OptiX buffers live as members of OPropagator in OptiX case, 
with OBuf jackets providing CBufSlice via slice method.


Migration from OpticksHub to Opticks for OpticksEvent supply 
----------------------------------------------------------------

OpticksRun.m_run is resident of Opticks (visitor to OpticksHub) 
so there is no reason to go up to OpticksHub level for OpticksEvent
access ... get your OpticksRun from Opticks and OpticksEvent 
from there : or just get OpticksEvent from Opticks (goes via OpticksRun)


**/

class Opticks ; 
class OEvent ; 
class OContext ; 
class TBuf ; 

struct CBufSpec ; 

#include "OKOP_API_EXPORT.hh"

class OKOP_API OpSeeder {
   public:
      OpSeeder(Opticks* ok, OEvent* oevt);
   public:
      void seedPhotonsFromGensteps();
   private:
      void seedComputeSeedsFromInteropGensteps();  // used WITH_SEED_BUFFER
      void seedPhotonsFromGenstepsViaOptiX();
      void seedPhotonsFromGenstepsViaOpenGL();
      void seedPhotonsFromGenstepsImp(const CBufSpec& rgs_, const CBufSpec& rox_);
      unsigned getNumPhotonsCheck(const TBuf& tgs);
   private:
      Opticks*                 m_ok ;
      bool                     m_dbg ; 
      OEvent*                  m_oevt ;
      OContext*                m_ocontext ;
};


