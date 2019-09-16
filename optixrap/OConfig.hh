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
OConfig
==========

OptiX utilities for raytrace program creation.


**/


#include <string>
#include <map>
#include <vector>

#include "plog/Severity.h"

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>

struct OProg ; 

#include "OXRAP_API_EXPORT.hh"
#include "OXRAP_HEAD.hh"

/**
OConfig
==========

ptxrel 
    relative directory beneath installcache/PTX, this defaults to nullptr 
    for PTX used only by tests set this to "tests"

**/


class OXRAP_API OConfig {

public:
  static const plog::Severity LEVEL ;  

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



public:
  // static const char* RngDir();

   static void Print(const char* msg="OConfig::Print");
   static optix::float3 make_contrast_color(int tag);
   static unsigned Multiplicity(RTformat format);
   static const char* FormatName(RTformat format);
   static void configureSampler(optix::TextureSampler& sampler, optix::Buffer& buffer);
   static unsigned OptiXVersion();
   //static bool DefaultWithTop();
public:
    OConfig(optix::Context context, const char* cmake_target="OptiXRap", const char* ptxrel=nullptr );
    void dump(const char* msg="OConfig::dump");

    optix::Program createProgram(const char* cu_name, const char* progname) ;

    unsigned int addEntry(const char* cu_name, const char* raygen, const char* exception, bool defer=false);
    unsigned int addRayGenerationProgram( const char* cu_name, const char* progname, bool defer=false);
    unsigned int addExceptionProgram( const char* cu_name, const char* progname, bool defer=false);

    void setMissProgram( unsigned int raytype , const char* cu_name, const char* progname, bool defer=false);
    void apply();
    void addProg(OProg* prog, bool defer);
    void apply(OProg* prog);
    unsigned int getNumEntryPoint();


private:
    friend struct rayleighTest ; 
    friend class interpolationTest ; 
    void setPTXRel(const char* ptxrel); 
    void setCMakeTarget(const char* cmake_target);
private:

    optix::Context m_context ;
    const char*  m_cmake_target ;  
    const char*  m_ptxrel ;  
    int          m_index_max ; 
    unsigned int m_raygen_index ;  
    unsigned int m_exception_index ;  

    std::map<std::string,optix::Program> m_programs;
    std::vector<OProg*> m_progs ; 

};

#include "OXRAP_TAIL.hh"



