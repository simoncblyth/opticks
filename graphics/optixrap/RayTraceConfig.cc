#include "RayTraceConfig.hh"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"

RayTraceConfig* RayTraceConfig::g_instance = NULL ;  
RayTraceConfig* RayTraceConfig::getInstance()
{
   return g_instance ; 
}

RayTraceConfig* RayTraceConfig::makeInstance(optix::Context context, const char* target)
{
   if(!g_instance)
   {  
      g_instance = new RayTraceConfig(context, target);
   }
   return g_instance ; 
}

const char* RayTraceConfig::SrcDir()
{
    return getenv("RAYTRACE_SRC_DIR") ;
} 

const char* RayTraceConfig::PtxDir()
{
    return getenv("RAYTRACE_PTX_DIR") ;
} 

const char* RayTraceConfig::RngDir()
{
    return getenv("RAYTRACE_RNG_DIR") ;
} 




void RayTraceConfig::Print(const char* msg)
{
    printf("%s \n", msg);
    printf("SrcDir %s \n", SrcDir() );
    printf("PtxDir %s \n", PtxDir() );
    printf("RngDir %s \n", RngDir() );
}


const char* const RayTraceConfig::ptxpath( const std::string& target, const std::string& base )
{
  static std::string path;
  path = std::string(PtxDir()) + "/" + target + "_generated_" + base + ".ptx";
  const char* _path = path.c_str();
  //printf("RayTraceConfig::ptxpath %s \n", _path); 
  return _path ; 
}



RayTraceConfig::RayTraceConfig(optix::Context context, const char* target )
        : 
        m_context(context),
        m_target(NULL)
{
    if(!target ) return ; 
    printf("RayTraceConfig::RayTraceConfig ctor ptxfold %s target %s  \n", PtxDir(), target );

    m_target = strdup(target);
}

RayTraceConfig::~RayTraceConfig(void)
{
    printf("RayTraceConfig dtor\n");
    free(m_target);
}



const char* const RayTraceConfig::ptxpath( const char* filename )
{
  static std::string path;
  path = ptxpath(m_target, filename);
  return path.c_str();
}


optix::Program RayTraceConfig::createProgram(const char* filename, const char* fname )
{
  std::string path = ptxpath(filename); 
  std::string key = path + ":" + fname ; 

  if(m_programs.find(key) == m_programs.end())
  { 
       printf("RayTraceConfig::createProgram key %s \n", key.c_str() );
       optix::Program program = m_context->createProgramFromPTXFile( path.c_str(), fname ); 
       m_programs[key] = program ; 
  } 
  else
  {
       //printf("createProgram cached key %s \n", key.c_str() );
  } 
  return m_programs[key];
}

void RayTraceConfig::setRayGenerationProgram( unsigned int index , const char* filename, const char* fname )
{
    optix::Program program = createProgram(filename, fname);
    m_context->setRayGenerationProgram( index, program ); 
}

void RayTraceConfig::setExceptionProgram( unsigned int index , const char* filename, const char* fname )
{
    optix::Program program = createProgram(filename, fname);
    m_context->setExceptionProgram( index, program ); 
}

void RayTraceConfig::setMissProgram( unsigned int index , const char* filename, const char* fname )
{
    optix::Program program = createProgram(filename, fname);
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










