#pragma once

#include "OProg.hh"
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <string>
#include <map>
#include <vector>

class OConfig {
public:
  // singleton 
  //static OConfig* g_instance ; 
  //static OConfig* getInstance();
  //static OConfig* makeInstance(optix::Context context);

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
   static const char* RngDir();

   static void Print(const char* msg="OConfig::Print");
   static optix::float3 make_contrast_color(int tag);
   static unsigned int getMultiplicity(RTformat format);
   static const char* getFormatName(RTformat format);

public:
    OConfig(optix::Context context);
    void dump(const char* msg="OConfig::dump");

    optix::Program createProgram(const char* filename, const char* progname );
    void setRayGenerationProgram( unsigned int index , const char* filename, const char* progname, bool defer=false);
    void setExceptionProgram( unsigned int index , const char* filename, const char* progname, bool defer=false);
    void setMissProgram( unsigned int index , const char* filename, const char* progname, bool defer=false);
    void apply();
    void addProg(OProg* prog, bool defer);
    void apply(OProg* prog);
    unsigned int getNumEntryPoint();

private:

    optix::Context m_context ;
    int          m_index_max ; 

    std::map<std::string,optix::Program> m_programs;
    std::vector<OProg*> m_progs ; 

};


inline OConfig::OConfig(optix::Context context )
        : 
        m_context(context),
        m_index_max(-1)
{
}




