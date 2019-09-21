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
OBufBase
==========

Wrapped OptiX GPU buffer, providing non-type specific utilities such 
as upload/download to NPY arrays. 


DevNotes
---------

* implementation in OBufBase_.cu as requires nvcc compilation

**/


#include "OXRAP_PUSH.hh"
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>
#include "OXRAP_POP.hh"

#include <string>

// cudawrap- struct 
#include "CBufSlice.hh"
#include "CBufSpec.hh"
class NPYBase ; 

// non-view-type specifics
#include "OXRAP_API_EXPORT.hh"

class OXRAP_API OBufBase {
   public:
      OBufBase( const char* name, optix::Buffer& buffer);
      virtual ~OBufBase();
   public:
      void upload(NPYBase* npy);
      void download(NPYBase* npy);
      void setHexDump(bool hexdump);
   private:
      void init();
      void examineBufferFormat(RTformat format);
      static unsigned long long NumBytes(const optix::Buffer& buffer);
   public:
      static unsigned long long Size(const optix::Buffer& buffer);
   public:
      CBufSpec  bufspec();
      CBufSlice slice( unsigned long long stride, unsigned long long begin=0ull , unsigned long long end=0ull );
      void*              getDevicePtr() ;
      unsigned long long getMultiplicity() const ; // typically 4, for RT_FORMAT_FLOAT4/RT_FORMAT_UINT4
      unsigned long long getSize() const ;         // width*depth*height of OptiX buffer, ie the number of typed elements (often float4) 
      unsigned long long getNumAtoms() const ;     // Multiplicity * Size, giving number of atoms, eg number of floats or ints
      unsigned long long getSizeOfAtom() const ;   // in bytes, eg 4 for any of the RT_FORMAT_FLOAT4 RT_FORMAT_FLOAT3 ... formats 
      unsigned long long getNumBytes() const ;     // total buffer size in bytes

   public:
      // usually set in ctor by examineBufferFormat, but RT_FORMAT_USER needs to be set manually 
      void setSizeOfAtom(unsigned long long soa);
      void setMultiplicity(unsigned long long mul);
   public:
      void Summary(const char* msg="OBufBase::Summary") const ;
      std::string desc() const ; 
   protected:
      optix::Buffer        m_buffer  ;
      const char*          m_name ; 
      unsigned long long   m_multiplicity ; 
      unsigned long long   m_sizeofatom ; 
      unsigned int         m_device ; 
      bool                 m_hexdump ; 
};


