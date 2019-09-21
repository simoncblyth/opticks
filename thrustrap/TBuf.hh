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

#include "CBufSpec.hh"
#include "CBufSlice.hh"

template <typename T> class NPY ; 

#include "THRAP_API_EXPORT.hh" 

/**
TBuf
=====

CUDA single buffer operations : upload/download/reductions/slicing/dumping

**/

class THRAP_API TBuf {
   public:
      TBuf(const char* name, CBufSpec spec, const char* delim=" \n" );
      void zero();

      void* getDevicePtr() const ;
      unsigned long long getNumBytes() const ;
      unsigned long long getSize() const ;      // NumItems might be better name
      unsigned long long getItemSize() const ;  //  NumBytes/Size

     
      unsigned downloadSelection4x4(const char* name, NPY<float>* npy, unsigned mskhis, bool verbose=false) const ; // selection done on items of size float4x4
      void dump4x4(const char* msg, unsigned long long stride, unsigned long long begin, unsigned long long end ) const ;

      template <typename T> unsigned downloadSelection(const char* name, NPY<float>* npy, unsigned mskhis, bool verbose=false) const ; // selection done on items of size T
      template <typename T> void fill(T value) const ;

      template <typename T> void upload(NPY<T>* npy) const ;
      template <typename T> void download(NPY<T>* npy, bool verbose=false) const ;

      template <typename T> void repeat_to(TBuf* other, unsigned long long stride, unsigned long long begin, unsigned long long end, unsigned long long repeats) const ;
      template <typename T> void dump(const char* msg, unsigned long long stride, unsigned long long begin, unsigned long long end ) const ;
      template <typename T> void dumpint(const char* msg, unsigned long long stride, unsigned long long begin, unsigned long long end ) const ;
      template <typename T> T  reduce(unsigned long long stride, unsigned long long begin, unsigned long long end=0ull ) const ;

      CBufSlice slice( unsigned long long stride, unsigned long long begin=0ull, unsigned long long end=0ull ) const ; 

      void Summary(const char* msg="TBuf::Summary") const ; 
   private:
      const char* m_name ;
      CBufSpec    m_spec ; 
      const char* m_delim ;
};


