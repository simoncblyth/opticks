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
OBuf
=====

Wrapper for OptiX buffers providing reduction and dumping 


DevNotes
-----------

* anything not using or related to the method template types should go into OBufBase

**/


#include "OBufBase.hh"


// NB implementation in OBuf_.cu as requires nvcc compilation

// TODO: avoid the code duplication between TBuf and OBuf ?
//  hmm OBuf could contain a TBuf ?
//  
//
//
// Using a templated class rather than templated member functions 
// has the advantage of only having to explicitly instanciate the class::
//
//    template class OBuf<optix::float4> ;
//    template class OBuf<optix::uint4> ;
//    template class OBuf<unsigned int> ;
//
// as opposed to having to explicly instanciate all the member functions.
//
// But when want differently typed "views" of the 
// same data it seems more logical to used templated member functions.
//

#include "OXRAP_API_EXPORT.hh"


class OXRAP_API OBuf : public OBufBase {
   public:
      OBuf( const char* name, optix::Buffer& buffer); 

   public:
      template <typename T>
      void dump(const char* msg, unsigned int stride, unsigned int begin, unsigned int end );

      template <typename T>
      void dumpint(const char* msg, unsigned int stride, unsigned int begin, unsigned int end );

      template <typename T>
      T reduce(unsigned int stride, unsigned int begin, unsigned int end=0u );

};


