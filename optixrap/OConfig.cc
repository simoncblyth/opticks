#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "BOpticksResource.hh"

#include "OProg.hh"
#include "OConfig.hh"

#include "PLOG.hh"



OConfig::OConfig(optix::Context context )
        : 
        m_context(context),
        m_index_max(-1),
        m_raygen_index(0),
        m_exception_index(0)
{
    
}

void OConfig::Print(const char* msg)
{
    printf("%s \n", msg);
}


unsigned OConfig::OptiXVersion()
{
   return OPTIX_VERSION ; 
}

bool OConfig::DefaultWithTop()
{
    unsigned version = OptiXVersion();
    bool with_top = false ; 
    switch(version)
    {
       case 3080:  with_top = false ;break; 
       case 3090:  with_top = false ;break; 
       case 40000: with_top = true ;break;      
       default: assert(0 && "unexpected OPTIX_VERSION") ;break;
    }

    if(version != 3080)
        LOG(warning) << "OConfig::DefaultWithTop"
                     << " proceeding with untested config " 
                     << " OPTIX_VERSION " << version 
                     << " with_top " << with_top
                     ;

    return with_top ;
}


optix::Program OConfig::createProgram(const char* filename, const char* progname )
{
  std::string path = BOpticksResource::PTXPath(filename); 
  std::string key = path + ":" + progname ; 

  LOG(trace) << "OConfig::createProgram"
             << " path " << path 
             ;

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



unsigned int OConfig::addEntry(const char* filename, const char* raygen, const char* exception, bool defer)
{
    int raygen_index = addRayGenerationProgram( filename, raygen, defer );
    int exception_index = addExceptionProgram( filename, exception, defer );
    assert(raygen_index == exception_index && raygen_index > -1);
    return raygen_index ; 
}


unsigned int OConfig::addRayGenerationProgram( const char* filename, const char* progname, bool defer)
{
    OProg* prog = new OProg('R', m_raygen_index, filename, progname);
    addProg(prog, defer);
    unsigned int index = m_raygen_index ;  
    m_raygen_index += 1 ;
    return index ; 
}

unsigned int OConfig::addExceptionProgram( const char* filename, const char* progname, bool defer)
{
    OProg* prog = new OProg('E', m_exception_index, filename, progname);
    addProg(prog, defer);
    unsigned int index = m_exception_index ;  
    m_exception_index += 1 ;
    return index ; 
}


void OConfig::setMissProgram( unsigned int raytype , const char* filename, const char* progname, bool defer)
{
    OProg* prog = new OProg('M', raytype, filename, progname);
    addProg(prog, defer);
}

void OConfig::addProg(OProg* prog, bool defer)
{
    int index = prog->index ; 

    LOG(trace) << "OConfig::addProg"
              << " desc " << prog->description()
              << " index/raytype " << index 
              ;

    m_progs.push_back(prog);
    if(!defer)
        apply(prog);
}
    
unsigned int OConfig::getNumEntryPoint()
{
    if(m_raygen_index != m_exception_index)
    {
        LOG(fatal) << "OConfig::getNumEntryPoint" 
                   << " EVERY RAYGEN PROGRAM NEEDS CORRESPONDING EXCEPTION PROGRAM " 
                   << " m_raygen_index " << m_raygen_index
                   << " m_exception_index " << m_exception_index
                 ;
    } 
    assert(m_raygen_index == m_exception_index);

    LOG(trace) << "OConfig::getNumEntryPoint" 
              << " m_raygen_index " << m_raygen_index
              << " m_exception_index " << m_exception_index
              ;

    return m_raygen_index ;  // already post incremented in the add  
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
              << " m_raygen_index " << m_raygen_index 
              << " m_exception_index " << m_exception_index 
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

#if OPTIX_VERSION > 3080
      case RT_FORMAT_HALF:  mul=1 ; break ;
      case RT_FORMAT_HALF2: mul=2 ; break ;
      case RT_FORMAT_HALF3: mul=3 ; break ;
      case RT_FORMAT_HALF4: mul=4 ; break ;
#endif

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

#if OPTIX_VERSION > 3080
      case RT_FORMAT_HALF:  name=_RT_FORMAT_HALF ; break ;
      case RT_FORMAT_HALF2: name=_RT_FORMAT_HALF2 ; break ;
      case RT_FORMAT_HALF3: name=_RT_FORMAT_HALF3 ; break ;
      case RT_FORMAT_HALF4: name=_RT_FORMAT_HALF4 ; break ;
#endif

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

#if OPTIX_VERSION > 3080
   const char* OConfig::_RT_FORMAT_HALF = "HALF" ;
   const char* OConfig::_RT_FORMAT_HALF2 = "HALF2" ;
   const char* OConfig::_RT_FORMAT_HALF3 = "HALF3" ;
   const char* OConfig::_RT_FORMAT_HALF4 = "HALF4" ;
#endif

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





void OConfig::configureSampler(optix::TextureSampler& sampler, optix::Buffer& buffer)
{
    LOG(trace) << "OPropertyLib::configureSampler" ; 

    // cuda-pdf p43 // default is to clamp to the range
    RTwrapmode wrapmode = RT_WRAP_REPEAT ;
    //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ;  // <--- seems not supported 
    //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
    //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ;  // return zero when out of range
    sampler->setWrapMode(0, wrapmode); 
    sampler->setWrapMode(1, wrapmode);

    //RTfiltermode filtermode = RT_FILTER_NEAREST ; 
    RTfiltermode filtermode = RT_FILTER_LINEAR ; 
    RTfiltermode minification = filtermode ; 
    RTfiltermode magnification = filtermode ; 
    RTfiltermode mipmapping = RT_FILTER_NONE ;

    sampler->setFilteringModes(minification, magnification, mipmapping);

    //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ;
    RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ;    // No conversion
    sampler->setReadMode(readmode);  

    //RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_ARRAY_INDEX ;  // by inspection : zero based array index offset by 0.5 (fails to validate in OptiX 400)
    RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ; 
    sampler->setIndexingMode(indexingmode);  


    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);        
    //   from 3.8 pdf: OptiX currently supports only a single MIP level and a single element texture array.

    unsigned int texture_array_idx = 0u ;
    unsigned int mip_level = 0u ; 
    sampler->setBuffer(texture_array_idx, mip_level, buffer);  // deprecated in OptiX 4

}


/*
(Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Unsupported combination of texture index, wrap and filter modes:  RT_TEXTURE_INDEX_ARRAY_INDEX, RT_WRAP_REPEAT, RT_FILTER_LINEAR, file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Util/TextureDescriptor.cpp, line: 138)


*/



/*


  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetWrapMode sets the wrapping mode of
  * \a texturesampler to \a wrapmode for the texture dimension specified
  * by \a dimension.  \a wrapmode can take one of the following values:
  *
  *  - @ref RT_WRAP_REPEAT
  *  - @ref RT_WRAP_CLAMP_TO_EDGE
  *  - @ref RT_WRAP_MIRROR
  *  - @ref RT_WRAP_CLAMP_TO_BORDER
  *
  * The wrapping mode controls the behavior of the texture sampler as
  * texture coordinates wrap around the range specified by the indexing
  * mode.  These values mirror the CUDA behavior of textures.
  * See CUDA programming guide for details.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   dimension        Dimension of the texture
  * @param[in]   wrapmode         The new wrap mode of the texture sampler



 * @ref rtTextureSamplerSetFilteringModes sets the minification, magnification and MIP mapping filter modes for \a texturesampler.
  * RTfiltermode must be one of the following values:
  *
  *  - @ref RT_FILTER_NEAREST
  *  - @ref RT_FILTER_LINEAR
  *  - @ref RT_FILTER_NONE
  *
  * These filter modes specify how the texture sampler will interpolate
  * buffer data that has been attached to it.  \a minification and
  * \a magnification must be one of @ref RT_FILTER_NEAREST or
  * @ref RT_FILTER_LINEAR.  \a mipmapping may be any of the three values but
  * must be @ref RT_FILTER_NONE if the texture sampler contains only a
  * single MIP level or one of @ref RT_FILTER_NEAREST or @ref RT_FILTER_LINEAR
  * if the texture sampler contains more than one MIP level.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   minification     The new minification filter mode of the texture sampler
  * @param[in]   magnification    The new magnification filter mode of the texture sampler
  * @param[in]   mipmapping       The new MIP mapping filter mode of the texture sampler
  *




 * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetIndexingMode sets the indexing mode of \a texturesampler to \a indexmode.  \a indexmode
  * can take on one of the following values:
  *
  *  - @ref RT_TEXTURE_INDEX_NORMALIZED_COORDINATES,
  *  - @ref RT_TEXTURE_INDEX_ARRAY_INDEX
  *
  * These values are used to control the interpretation of texture coordinates.  If the index mode is set to
  * @ref RT_TEXTURE_INDEX_NORMALIZED_COORDINATES, the texture is parameterized over [0,1].  If the index
  * mode is set to @ref RT_TEXTURE_INDEX_ARRAY_INDEX then texture coordinates are interpreted as array indices
  * into the contents of the underlying buffer objects.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   indexmode        The new indexing mode of the texture sampler
  *




*/




/*

OptiX 400

2016-08-10 15:05:35.709 INFO  [1903430] [OContext::launch@214] OContext::launch
entry 0 width 1 height 1 libc++abi.dylib: terminating with uncaught exception
of type optix::Exception: Invalid value (Details: Function "RTresult
_rtContextValidate(RTcontext)" caught exception: Unsupported combination of
texture index, wrap and filter modes:  RT_TEXTURE_INDEX_ARRAY_INDEX,
RT_WRAP_REPEAT, RT_FILTER_LINEAR,
file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Util/TextureDescriptor.cpp,
line: 138) Abort trap: 6

*/







