#pragma once
#define RAYTRACECONFIG_H

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <string>
#include <map>

class OConfig {

public:
   // singleton 
   static OConfig* g_instance ; 
   static OConfig* getInstance();
   static OConfig* makeInstance(optix::Context context);

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
   //static const char* SrcDir();
   //static const char* PtxDir();
   static const char* RngDir();

   static void Print(const char* msg="OConfig::Print");
   //static const char* const ptxpath( const std::string& target, const std::string& base );

   static optix::float3 make_contrast_color(int tag);
   static unsigned int getMultiplicity(RTformat format);
   static const char* getFormatName(RTformat format);

public:
    OConfig(optix::Context context);
    virtual ~OConfig();

    //const char* const ptxpath( const char* filename );
    optix::Program createProgram(const char* filename, const char* progname );

    void setRayGenerationProgram( unsigned int index , const char* filename, const char* progname );
    void setExceptionProgram( unsigned int index , const char* filename, const char* progname );
    void setMissProgram( unsigned int index , const char* filename, const char* progname );

private:
    optix::Context m_context ;

    //char* m_target ;

    std::map<std::string,optix::Program> m_programs;


};

