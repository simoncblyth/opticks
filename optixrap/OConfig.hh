#pragma once

/**
OConfig
==========

OptiX utilities for raytrace program creation.


**/


#include <string>
#include <map>
#include <vector>

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>

struct OProg ; 

#include "OXRAP_API_EXPORT.hh"
#include "OXRAP_HEAD.hh"

class OXRAP_API OConfig {
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


public:
  // static const char* RngDir();

   static void Print(const char* msg="OConfig::Print");
   static optix::float3 make_contrast_color(int tag);
   static unsigned int getMultiplicity(RTformat format);
   static const char* getFormatName(RTformat format);
   static void configureSampler(optix::TextureSampler& sampler, optix::Buffer& buffer);
   static unsigned OptiXVersion();
   static bool DefaultWithTop();
public:
    OConfig(optix::Context context);
    void dump(const char* msg="OConfig::dump");

    optix::Program createProgram(const char* filename, const char* progname );

    unsigned int addEntry(const char* filename, const char* raygen, const char* exception, bool defer=false);
    unsigned int addRayGenerationProgram( const char* filename, const char* progname, bool defer=false);
    unsigned int addExceptionProgram( const char* filename, const char* progname, bool defer=false);

    void setMissProgram( unsigned int raytype , const char* filename, const char* progname, bool defer=false);
    void apply();
    void addProg(OProg* prog, bool defer);
    void apply(OProg* prog);
    unsigned int getNumEntryPoint();

private:

    optix::Context m_context ;
    int          m_index_max ; 
    unsigned int m_raygen_index ;  
    unsigned int m_exception_index ;  

    std::map<std::string,optix::Program> m_programs;
    std::vector<OProg*> m_progs ; 

};

#include "OXRAP_TAIL.hh"



