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

// see /Developer/NVIDIA/CUDA-5.5/include/vector_types.h for types with builtin handling

// "standard" sized vector types, all 4*32 = 128 bit  (16 bytes)
union quad
{
   float4 f ;
   int4   i ;
   uint4  u ;
};


// "half" sized vector types, all 4*16 = 64 bit       (8 bytes)
union hquad
{
   short4   short_ ;
   ushort4  ushort_ ;
};


// "quarter" sized vector types, all 4*8 = 32 bit   (4 bytes)
union qquad
{
   char4   char_   ;
   uchar4  uchar_  ;
};

union uifchar4
{
   unsigned int u ; 
   int          i ; 
   float        f ; 
   char4        char_   ;
   uchar4       uchar_  ;
};

union uif
{
   unsigned int u ; 
   int          i ; 
   float        f ; 
};


// up to 16 signed char into 128bits of the uint4 
union uint4c16 
{
    uint4 u ; 
    signed char c[16] ; 
};








#ifdef __CUDACC__

static __device__
float unsigned_as_float(unsigned u)
{
  union {
    float f;
    unsigned u;
  } v1;

  v1.u = u; 
  return v1.f;
}


static __device__
float qint_as_float(int i)
{
  union {
    float f;
    int   i;
  } v1;

  v1.i = i; 
  return v1.f;
}


#endif




