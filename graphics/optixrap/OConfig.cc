#include "OConfig.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include "Config.hh"  // ptxpath, RNGDIR


#include "string.h"
#include "stdio.h"
#include "stdlib.h"


/*
OConfig* OConfig::g_instance = NULL ;  
OConfig* OConfig::getInstance()
{
   return g_instance ; 
}

OConfig* OConfig::makeInstance(optix::Context context)
{
   if(!g_instance)
   {  
      g_instance = new OConfig(context);
   }
   return g_instance ; 
}
*/


const char* OConfig::RngDir()
{
    return RNGDIR ; 
} 



void OConfig::Print(const char* msg)
{
    printf("%s \n", msg);
}

      

optix::Program OConfig::createProgram(const char* filename, const char* progname )
{
  std::string path = ptxpath(filename); 
  std::string key = path + ":" + progname ; 

  if(m_programs.find(key) == m_programs.end())
  { 
       LOG(debug) << "OConfig::createProgram " << key ;
       optix::Program program = m_context->createProgramFromPTXFile( path.c_str(), progname ); 
       m_programs[key] = program ; 
  } 
  else
  {
       //printf("createProgram cached key %s \n", key.c_str() );
  } 
  return m_programs[key];
}



void OConfig::setRayGenerationProgram( unsigned int index , const char* filename, const char* progname, bool defer)
{
    OProg* prog = new OProg('R', index, filename, progname);
    addProg(prog, defer);
}
void OConfig::setExceptionProgram( unsigned int index , const char* filename, const char* progname, bool defer)
{
    OProg* prog = new OProg('E', index, filename, progname);
    addProg(prog, defer);
}
void OConfig::setMissProgram( unsigned int index , const char* filename, const char* progname, bool defer)
{
    OProg* prog = new OProg('M', index, filename, progname);
    addProg(prog, defer);
}

void OConfig::addProg(OProg* prog, bool defer)
{
    int index = prog->index ; 
    if(index > m_index_max) m_index_max = index ;

    LOG(debug) << "OConfig::addProg"
              << " desc " << prog->description()
              << " index " << index 
              << " m_index_max " << m_index_max 
              ;

    m_progs.push_back(prog);
    if(!defer)
        apply(prog);
}
    
unsigned int OConfig::getNumEntryPoint()
{
    return m_index_max == -1 ? 0 : m_index_max + 1 ;  
}

void OConfig::apply(OProg* prog)
{
    unsigned int index = prog->index ; 
    char type = prog->type ; 
    optix::Program program = createProgram(prog->filename, prog->progname);
    switch(type)
    {
        case 'R':
            m_context->setRayGenerationProgram( index, program ); 
            break;
        case 'E':
            m_context->setExceptionProgram( index, program ); 
            break;
        case 'M':
            m_context->setMissProgram( index, program ); 
            break;
    }  
}

void OConfig::apply()
{
    for(unsigned int i=0 ; i < m_progs.size() ; i++)
    {
         OProg* prog = m_progs[i];
         apply(prog);
    }
}


void OConfig::dump(const char* msg)
{
    LOG(info) << msg 
              << " m_index_max " << m_index_max 
              ;
    for(unsigned int i=0 ; i < m_progs.size() ; i++)
    {
         OProg* prog = m_progs[i];
         printf("%s\n", prog->description() );
    }
}






optix::float3 OConfig::make_contrast_color( int tag )
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



unsigned int OConfig::getMultiplicity(RTformat format)
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




const char* OConfig::getFormatName(RTformat format)
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




   const char* OConfig::_RT_FORMAT_UNKNOWN = "UNKNOWN" ;

   const char* OConfig::_RT_FORMAT_FLOAT = "FLOAT" ;
   const char* OConfig::_RT_FORMAT_FLOAT2 = "FLOAT2" ;
   const char* OConfig::_RT_FORMAT_FLOAT3 = "FLOAT3" ;
   const char* OConfig::_RT_FORMAT_FLOAT4 = "FLOAT4" ;

   const char* OConfig::_RT_FORMAT_BYTE = "BYTE" ;
   const char* OConfig::_RT_FORMAT_BYTE2 = "BYTE2" ;
   const char* OConfig::_RT_FORMAT_BYTE3 = "BYTE3" ;
   const char* OConfig::_RT_FORMAT_BYTE4 = "BYTE4" ;

   const char* OConfig::_RT_FORMAT_UNSIGNED_BYTE = "UNSIGNED_BYTE" ;
   const char* OConfig::_RT_FORMAT_UNSIGNED_BYTE2 = "UNSIGNED_BYTE2" ;
   const char* OConfig::_RT_FORMAT_UNSIGNED_BYTE3 = "UNSIGNED_BYTE3" ;
   const char* OConfig::_RT_FORMAT_UNSIGNED_BYTE4 = "UNSIGNED_BYTE4" ;

   const char* OConfig::_RT_FORMAT_SHORT = "SHORT" ;
   const char* OConfig::_RT_FORMAT_SHORT2 = "SHORT2" ;
   const char* OConfig::_RT_FORMAT_SHORT3 = "SHORT3" ;
   const char* OConfig::_RT_FORMAT_SHORT4 = "SHORT4" ;

   const char* OConfig::_RT_FORMAT_UNSIGNED_SHORT = "UNSIGNED_SHORT" ;
   const char* OConfig::_RT_FORMAT_UNSIGNED_SHORT2 = "UNSIGNED_SHORT2" ;
   const char* OConfig::_RT_FORMAT_UNSIGNED_SHORT3 = "UNSIGNED_SHORT3";
   const char* OConfig::_RT_FORMAT_UNSIGNED_SHORT4 = "UNSIGNED_SHORT4";

   const char* OConfig::_RT_FORMAT_INT = "INT" ;
   const char* OConfig::_RT_FORMAT_INT2 = "INT2";
   const char* OConfig::_RT_FORMAT_INT3 = "INT3";
   const char* OConfig::_RT_FORMAT_INT4 = "INT4";

   const char* OConfig::_RT_FORMAT_UNSIGNED_INT = "UNSIGNED_INT" ;
   const char* OConfig::_RT_FORMAT_UNSIGNED_INT2 = "UNSIGNED_INT2";
   const char* OConfig::_RT_FORMAT_UNSIGNED_INT3 = "UNSIGNED_INT3";
   const char* OConfig::_RT_FORMAT_UNSIGNED_INT4 = "UNSIGNED_INT4";

   const char* OConfig::_RT_FORMAT_USER = "USER" ;
   const char* OConfig::_RT_FORMAT_BUFFER_ID = "BUFFER_ID" ;
   const char* OConfig::_RT_FORMAT_PROGRAM_ID = "PROGRAM_ID" ;


