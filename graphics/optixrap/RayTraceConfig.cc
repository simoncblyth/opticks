#include "RayTraceConfig.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include "Config.hh"  // ptxpath, RNGDIR


#include "string.h"
#include "stdio.h"
#include "stdlib.h"

RayTraceConfig* RayTraceConfig::g_instance = NULL ;  
RayTraceConfig* RayTraceConfig::getInstance()
{
   return g_instance ; 
}

RayTraceConfig* RayTraceConfig::makeInstance(optix::Context context)
{
   if(!g_instance)
   {  
      g_instance = new RayTraceConfig(context);
   }
   return g_instance ; 
}


/*
const char* RayTraceConfig::SrcDir()
{
    return getenv("RAYTRACE_SRC_DIR") ;
} 

const char* RayTraceConfig::PtxDir()
{
    return getenv("RAYTRACE_PTX_DIR") ;
} 
*/

const char* RayTraceConfig::RngDir()
{
    //return getenv("RAYTRACE_RNG_DIR") ;
    return RNGDIR ; 
} 



void RayTraceConfig::Print(const char* msg)
{
    printf("%s \n", msg);
/*
    printf("PtxDir %s \n", PtxDir() );
    printf("RngDir %s \n", RngDir() );
*/
}


RayTraceConfig::RayTraceConfig(optix::Context context )
        : 
        m_context(context)
{
   // LOG(debug) << "RayTraceConfig::RayTraceConfig ptxdir " << PtxDir() << " target " <<  target ;
   // m_target = strdup(target);
}

RayTraceConfig::~RayTraceConfig(void)
{
}



optix::Program RayTraceConfig::createProgram(const char* filename, const char* progname )
{
  std::string path = ptxpath(filename); 
  std::string key = path + ":" + progname ; 

  if(m_programs.find(key) == m_programs.end())
  { 
       LOG(debug) << "RayTraceConfig::createProgram " << key ;
       optix::Program program = m_context->createProgramFromPTXFile( path.c_str(), progname ); 
       m_programs[key] = program ; 
  } 
  else
  {
       //printf("createProgram cached key %s \n", key.c_str() );
  } 
  return m_programs[key];
}

void RayTraceConfig::setRayGenerationProgram( unsigned int index , const char* filename, const char* progname )
{
    optix::Program program = createProgram(filename, progname);
    m_context->setRayGenerationProgram( index, program ); 
}

void RayTraceConfig::setExceptionProgram( unsigned int index , const char* filename, const char* progname )
{
    optix::Program program = createProgram(filename, progname);
    m_context->setExceptionProgram( index, program ); 
}

void RayTraceConfig::setMissProgram( unsigned int index , const char* filename, const char* progname )
{
    optix::Program program = createProgram(filename, progname);
    m_context->setMissProgram( index, program ); 
}




optix::float3 RayTraceConfig::make_contrast_color( int tag )
{
  static const unsigned char s_Colors[16][3] =
  {
    {  34, 139,  34}, // ForestGreen
    { 210, 180, 140}, // Tan
    { 250, 128, 114}, // Salmon
    { 173, 255,  47}, // GreenYellow
    { 255,   0, 255}, // Magenta
    { 255,   0,   0}, // Red
    {   0, 250, 154}, // MediumSpringGreen
    { 255, 165,   0}, // Orange
    { 240, 230, 140}, // Khaki
    { 255, 215,   0}, // Gold
    { 178,  34,  34}, // Firebrick
    { 154, 205,  50}, // YellowGreen
    {  64, 224, 208}, // Turquoise
    {   0,   0, 255}, // Blue
    { 100, 149, 237}, // CornflowerBlue
    { 153, 153, 255}, // (bright blue)
  };  
  int i = tag & 0x0f;
  optix::float3 color = optix::make_float3( s_Colors[i][0], s_Colors[i][1], s_Colors[i][2] );
  color *= 1.f/255.f;
  i = (tag >> 4) & 0x3;
  color *= 1.f - float(i) * 0.23f;
  return color;
}    



unsigned int RayTraceConfig::getMultiplicity(RTformat format)
{
   unsigned int mul(0) ;
   switch(format)
   {
      case RT_FORMAT_UNKNOWN: mul=0 ; break ; 

      case RT_FORMAT_FLOAT:   mul=1 ; break ;
      case RT_FORMAT_FLOAT2:  mul=2 ; break ;
      case RT_FORMAT_FLOAT3:  mul=3 ; break ;
      case RT_FORMAT_FLOAT4:  mul=4 ; break ;

      case RT_FORMAT_BYTE:    mul=1 ; break ;
      case RT_FORMAT_BYTE2:   mul=2 ; break ;
      case RT_FORMAT_BYTE3:   mul=3 ; break ;
      case RT_FORMAT_BYTE4:   mul=4 ; break ;

      case RT_FORMAT_UNSIGNED_BYTE:  mul=1 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE2: mul=2 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE3: mul=3 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE4: mul=4 ; break ;

      case RT_FORMAT_SHORT:  mul=1 ; break ;
      case RT_FORMAT_SHORT2: mul=2 ; break ;
      case RT_FORMAT_SHORT3: mul=3 ; break ;
      case RT_FORMAT_SHORT4: mul=4 ; break ;

      case RT_FORMAT_UNSIGNED_SHORT:  mul=1 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT2: mul=2 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT3: mul=3 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT4: mul=4 ; break ;

      case RT_FORMAT_INT:  mul=1 ; break ;
      case RT_FORMAT_INT2: mul=2 ; break ;
      case RT_FORMAT_INT3: mul=3 ; break ;
      case RT_FORMAT_INT4: mul=4 ; break ;

      case RT_FORMAT_UNSIGNED_INT:  mul=1 ; break ;
      case RT_FORMAT_UNSIGNED_INT2: mul=2 ; break ;
      case RT_FORMAT_UNSIGNED_INT3: mul=3 ; break ;
      case RT_FORMAT_UNSIGNED_INT4: mul=4 ; break ;

      case RT_FORMAT_USER:       mul=0 ; break ;
      case RT_FORMAT_BUFFER_ID:  mul=0 ; break ;
      case RT_FORMAT_PROGRAM_ID: mul=0 ; break ; 
   }
   return mul ; 
}




const char* RayTraceConfig::getFormatName(RTformat format)
{
   const char* name = NULL ; 
   switch(format)
   {
      case RT_FORMAT_UNKNOWN: name=_RT_FORMAT_UNKNOWN ; break ; 

      case RT_FORMAT_FLOAT:   name=_RT_FORMAT_FLOAT ; break ;
      case RT_FORMAT_FLOAT2:  name=_RT_FORMAT_FLOAT2 ; break ;
      case RT_FORMAT_FLOAT3:  name=_RT_FORMAT_FLOAT3 ; break ;
      case RT_FORMAT_FLOAT4:  name=_RT_FORMAT_FLOAT4 ; break ;

      case RT_FORMAT_BYTE:    name=_RT_FORMAT_BYTE ; break ;
      case RT_FORMAT_BYTE2:   name=_RT_FORMAT_BYTE2 ; break ;
      case RT_FORMAT_BYTE3:   name=_RT_FORMAT_BYTE3 ; break ;
      case RT_FORMAT_BYTE4:   name=_RT_FORMAT_BYTE4 ; break ;

      case RT_FORMAT_UNSIGNED_BYTE:  name=_RT_FORMAT_UNSIGNED_BYTE ; break ;
      case RT_FORMAT_UNSIGNED_BYTE2: name=_RT_FORMAT_UNSIGNED_BYTE2 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE3: name=_RT_FORMAT_UNSIGNED_BYTE3 ; break ;
      case RT_FORMAT_UNSIGNED_BYTE4: name=_RT_FORMAT_UNSIGNED_BYTE4 ; break ;

      case RT_FORMAT_SHORT:  name=_RT_FORMAT_SHORT ; break ;
      case RT_FORMAT_SHORT2: name=_RT_FORMAT_SHORT2 ; break ;
      case RT_FORMAT_SHORT3: name=_RT_FORMAT_SHORT3 ; break ;
      case RT_FORMAT_SHORT4: name=_RT_FORMAT_SHORT4 ; break ;

      case RT_FORMAT_UNSIGNED_SHORT:  name=_RT_FORMAT_UNSIGNED_SHORT ; break ;
      case RT_FORMAT_UNSIGNED_SHORT2: name=_RT_FORMAT_UNSIGNED_SHORT2 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT3: name=_RT_FORMAT_UNSIGNED_SHORT3 ; break ;
      case RT_FORMAT_UNSIGNED_SHORT4: name=_RT_FORMAT_UNSIGNED_SHORT4 ; break ;

      case RT_FORMAT_INT:  name=_RT_FORMAT_INT ; break ;
      case RT_FORMAT_INT2: name=_RT_FORMAT_INT2 ; break ;
      case RT_FORMAT_INT3: name=_RT_FORMAT_INT3 ; break ;
      case RT_FORMAT_INT4: name=_RT_FORMAT_INT4 ; break ;

      case RT_FORMAT_UNSIGNED_INT:  name=_RT_FORMAT_UNSIGNED_INT ; break ;
      case RT_FORMAT_UNSIGNED_INT2: name=_RT_FORMAT_UNSIGNED_INT2 ; break ;
      case RT_FORMAT_UNSIGNED_INT3: name=_RT_FORMAT_UNSIGNED_INT3 ; break ;
      case RT_FORMAT_UNSIGNED_INT4: name=_RT_FORMAT_UNSIGNED_INT4 ; break ;

      case RT_FORMAT_USER:       name=_RT_FORMAT_USER ; break ;
      case RT_FORMAT_BUFFER_ID:  name=_RT_FORMAT_BUFFER_ID ; break ;
      case RT_FORMAT_PROGRAM_ID: name=_RT_FORMAT_PROGRAM_ID ; break ; 
   }
   return name ; 
}




   const char* RayTraceConfig::_RT_FORMAT_UNKNOWN = "UNKNOWN" ;

   const char* RayTraceConfig::_RT_FORMAT_FLOAT = "FLOAT" ;
   const char* RayTraceConfig::_RT_FORMAT_FLOAT2 = "FLOAT2" ;
   const char* RayTraceConfig::_RT_FORMAT_FLOAT3 = "FLOAT3" ;
   const char* RayTraceConfig::_RT_FORMAT_FLOAT4 = "FLOAT4" ;

   const char* RayTraceConfig::_RT_FORMAT_BYTE = "BYTE" ;
   const char* RayTraceConfig::_RT_FORMAT_BYTE2 = "BYTE2" ;
   const char* RayTraceConfig::_RT_FORMAT_BYTE3 = "BYTE3" ;
   const char* RayTraceConfig::_RT_FORMAT_BYTE4 = "BYTE4" ;

   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_BYTE = "UNSIGNED_BYTE" ;
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_BYTE2 = "UNSIGNED_BYTE2" ;
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_BYTE3 = "UNSIGNED_BYTE3" ;
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_BYTE4 = "UNSIGNED_BYTE4" ;

   const char* RayTraceConfig::_RT_FORMAT_SHORT = "SHORT" ;
   const char* RayTraceConfig::_RT_FORMAT_SHORT2 = "SHORT2" ;
   const char* RayTraceConfig::_RT_FORMAT_SHORT3 = "SHORT3" ;
   const char* RayTraceConfig::_RT_FORMAT_SHORT4 = "SHORT4" ;

   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_SHORT = "UNSIGNED_SHORT" ;
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_SHORT2 = "UNSIGNED_SHORT2" ;
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_SHORT3 = "UNSIGNED_SHORT3";
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_SHORT4 = "UNSIGNED_SHORT4";

   const char* RayTraceConfig::_RT_FORMAT_INT = "INT" ;
   const char* RayTraceConfig::_RT_FORMAT_INT2 = "INT2";
   const char* RayTraceConfig::_RT_FORMAT_INT3 = "INT3";
   const char* RayTraceConfig::_RT_FORMAT_INT4 = "INT4";

   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_INT = "UNSIGNED_INT" ;
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_INT2 = "UNSIGNED_INT2";
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_INT3 = "UNSIGNED_INT3";
   const char* RayTraceConfig::_RT_FORMAT_UNSIGNED_INT4 = "UNSIGNED_INT4";

   const char* RayTraceConfig::_RT_FORMAT_USER = "USER" ;
   const char* RayTraceConfig::_RT_FORMAT_BUFFER_ID = "BUFFER_ID" ;
   const char* RayTraceConfig::_RT_FORMAT_PROGRAM_ID = "PROGRAM_ID" ;


