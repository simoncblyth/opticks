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
OFormat
==========

**/

#include "OXPPNS.hh"

#include "OXRAP_API_EXPORT.hh"
#include "OXRAP_HEAD.hh"

class OXRAP_API OFormat {
public:
  static unsigned long long ElementSizeInBytes(RTformat format); // eg sizeof(RT_FORMAT_FLOAT4) = 4*4 = 16 
  static unsigned long long Multiplicity(RTformat format);
  static const char* FormatName(RTformat format);
public:
  static const char* _RT_FORMAT_UNKNOWN;

  static const char* _RT_FORMAT_FLOAT;
  static const char* _RT_FORMAT_FLOAT2;
  static const char* _RT_FORMAT_FLOAT3;
  static const char* _RT_FORMAT_FLOAT4;

  static const char* _RT_FORMAT_BYTE;
  static const char* _RT_FORMAT_BYTE2;
  static const char* _RT_FORMAT_BYTE3;
  static const char* _RT_FORMAT_BYTE4;

  static const char* _RT_FORMAT_UNSIGNED_BYTE;
  static const char* _RT_FORMAT_UNSIGNED_BYTE2;
  static const char* _RT_FORMAT_UNSIGNED_BYTE3;
  static const char* _RT_FORMAT_UNSIGNED_BYTE4;

  static const char* _RT_FORMAT_SHORT;
  static const char* _RT_FORMAT_SHORT2;
  static const char* _RT_FORMAT_SHORT3;
  static const char* _RT_FORMAT_SHORT4;

#if OPTIX_VERSION >= 400
  static const char* _RT_FORMAT_HALF;
  static const char* _RT_FORMAT_HALF2;
  static const char* _RT_FORMAT_HALF3;
  static const char* _RT_FORMAT_HALF4;
#endif

  static const char* _RT_FORMAT_UNSIGNED_SHORT;
  static const char* _RT_FORMAT_UNSIGNED_SHORT2;
  static const char* _RT_FORMAT_UNSIGNED_SHORT3;
  static const char* _RT_FORMAT_UNSIGNED_SHORT4;

  static const char* _RT_FORMAT_INT;
  static const char* _RT_FORMAT_INT2;
  static const char* _RT_FORMAT_INT3;
  static const char* _RT_FORMAT_INT4;

  static const char* _RT_FORMAT_UNSIGNED_INT;
  static const char* _RT_FORMAT_UNSIGNED_INT2;
  static const char* _RT_FORMAT_UNSIGNED_INT3;
  static const char* _RT_FORMAT_UNSIGNED_INT4;

  static const char* _RT_FORMAT_USER;
  static const char* _RT_FORMAT_BUFFER_ID;
  static const char* _RT_FORMAT_PROGRAM_ID;

#if OPTIX_VERSION >= 60000
  static const char* _RT_FORMAT_LONG_LONG ;  
  static const char* _RT_FORMAT_LONG_LONG2 ; 
  static const char* _RT_FORMAT_LONG_LONG3 ; 
  static const char* _RT_FORMAT_LONG_LONG4 ; 

  static const char* _RT_FORMAT_UNSIGNED_LONG_LONG  ; 
  static const char* _RT_FORMAT_UNSIGNED_LONG_LONG2 ; 
  static const char* _RT_FORMAT_UNSIGNED_LONG_LONG3 ; 
  static const char* _RT_FORMAT_UNSIGNED_LONG_LONG4 ; 
   
  static const char* _RT_FORMAT_UNSIGNED_BC1 ; 
  static const char* _RT_FORMAT_UNSIGNED_BC2 ; 
  static const char* _RT_FORMAT_UNSIGNED_BC3 ; 
  static const char* _RT_FORMAT_UNSIGNED_BC4 ; 
  static const char* _RT_FORMAT_UNSIGNED_BC5 ; 
  static const char* _RT_FORMAT_UNSIGNED_BC6H ; 
  static const char* _RT_FORMAT_UNSIGNED_BC7 ; 

  static const char* _RT_FORMAT_BC4 ; 
  static const char* _RT_FORMAT_BC5 ; 
  static const char* _RT_FORMAT_BC6H ; 
#endif

};

#include "OXRAP_TAIL.hh"



