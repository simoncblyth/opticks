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
TSparse<T>
============

Utilities for rudimentatry GPU integer histogramming, used for 
finding most frequent photon histories. 

**/


#include <string>

#include "CBufSlice.hh"
#include "THRAP_PUSH.hh"
#include <thrust/device_vector.h>
#include "THRAP_POP.hh"


class Index ; 

#define TSPARSE_LOOKUP_N 32

#include "THRAP_API_EXPORT.hh"

/**
TSparse
=========

Used for photon history/material sequence indexing, yielding 
a history/material sequence popularity ranking for every photon. 

These rankings are used from the OpenGL geometry shaders such as
oglrap/gl/rec/geom.glsl to provide interactive selection in the 
display of photons records to render.

**/


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
      const char* m_reldir ; 
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


