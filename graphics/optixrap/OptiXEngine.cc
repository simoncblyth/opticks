#include "OptiXEngine.hh"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include <optixu/optixu.h>
//#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

// npy-
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NumpyEvt.hpp"


// oglrap-
#include "Composition.hh"
#include "Renderer.hh"
#include "Texture.hh"

// ggeo-
#include "GGeo.hh"
#include "GMergedMesh.hh"

// optixrap-
#include "RayTraceConfig.hh"
#include "GGeoOptiXGeometry.hh"


#include "assert.h"
#include "stdio.h"
#include "string.h"

using namespace optix;

enum RayType
{
   radiance_ray_type,
   shadow_ray_type
};


// extracts from /usr/local/env/cuda/OptiX_370b2_sdk/sutil/SampleScene.cpp

OptiXEngine::OptiXEngine(const char* cmake_target) :
    m_context(NULL),
    m_geometry_group(NULL),
    m_photon_buffer_id(0),
    m_genstep_buffer_id(0),
    m_pbo(0),
    m_pbo_data(NULL),
    m_composition(NULL),
    m_renderer(NULL),
    m_texture(NULL),
    m_config(NULL),
    m_ggeo(NULL),
    m_mergedmesh(NULL),
    m_trace_count(0),
    m_cmake_target(strdup(cmake_target)),
    m_enabled(true),
    m_texture_id(-1),
    m_filename(),
    m_accel_cache_loaded(false),
    m_accel_caching_on(true)
{
    LOG(info) << "OptiXEngine::OptiXEngine" ;
}


void OptiXEngine::init()
{
    if(!m_enabled) return ;

    LOG(info) << "OptiXEngine::init " ;
    m_context = Context::create();
    m_geometry_group = m_context->createGeometryGroup();
    m_config = RayTraceConfig::makeInstance(m_context, m_cmake_target);

    initRenderer();
    initContext();
    initGeometry();
    preprocess();

    LOG(info) << "OptiXEngine::init DONE " ;
}

void OptiXEngine::initRenderer()
{
    unsigned int width  = m_composition->getWidth();
    unsigned int height = m_composition->getHeight();

    m_renderer = new Renderer("tex");
    m_texture = new Texture();   // QuadTexture would be better name
    m_texture->setSize(width, height);
    m_texture->create();
    m_texture_id = m_texture->getTextureId() ;

    LOG(info) << "OptiXEngine::initRenderer size(" << width << "," << height << ")  texture_id " << m_texture_id ;
    m_renderer->setDrawable(m_texture);
}

void OptiXEngine::initContext()
{
    unsigned int width  = m_composition->getWidth();
    unsigned int height = m_composition->getHeight();

    LOG(info) << "OptiXEngine::initContext size (" << width << "," << height << ")" ;

    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setPrintLaunchIndex(0,0,0);

    m_context->setStackSize( 2180 );
 
    m_output_buffer = createOutputBuffer_PBO(m_pbo, RT_FORMAT_UNSIGNED_BYTE4, width, height) ;
    m_context["output_buffer"]->set( m_output_buffer );

    m_context["touch_buffer"]->set( m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 1, 1));


    unsigned int num_entry_points = 1;
    m_context->setEntryPointCount( num_entry_points );

    RayTraceConfig* cfg = RayTraceConfig::getInstance();

    unsigned int entry_point_index = 0 ; 
    cfg->setRayGenerationProgram(entry_point_index, "pinhole_camera.cu", "pinhole_camera" );

    cfg->setExceptionProgram(entry_point_index, "pinhole_camera.cu", "exception");
    m_context[ "bad_color" ]->setFloat( 1.0f, 0.0f, 0.0f );
    m_context[ "radiance_ray_type"   ]->setUint( radiance_ray_type );

    unsigned int num_ray_types = 2 ; 
    m_context->setRayTypeCount( num_ray_types );

    unsigned int ray_type_index = 0 ; 
    cfg->setMissProgram(ray_type_index, "constantbg.cu", "miss" );
    m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f ); // map(int,np.array([0.34,0.55,0.85])*255) -> [86, 140, 216]
}

void OptiXEngine::initGeometry()
{
    LOG(info) << "OptiXEngine::initGeometry" ;
    // TODO: use GMergedMesh instead of GGeo
    //GGeoOptiXGeometry geom(m_ggeo);
    GGeoOptiXGeometry geom(m_ggeo, m_mergedmesh);
    
    geom.setGeometryGroup(m_geometry_group);
    geom.setContext(m_context);   
    //geom.setOverrideMaterial(m_material);  

    geom.convert(); 

    setFilename(m_ggeo->getIdentityPath());

    geom.setupAcceleration();

    loadAccelCache();

    m_aabb = geom.getAabb();

    m_context[ "top_object" ]->set( m_geometry_group );

    // cf with MeshViewer::initGeometry
    LOG(info) << "OptiXEngine::initGeometry DONE " ;
}


void OptiXEngine::preprocess()
{
    LOG(info)<< "OptiXEngine::preprocess";

    m_context[ "scene_epsilon"]->setFloat(m_composition->getNear());
 
    LOG(info)<< "OptiXEngine::preprocess start validate ";
    m_context->validate();
    LOG(info)<< "OptiXEngine::preprocess start compile ";
    m_context->compile();
    LOG(info)<< "OptiXEngine::preprocess start building Accel structure ";
    m_context->launch(0,0); 

    LOG(info)<< "OptiXEngine::preprocess DONE ";
}

void OptiXEngine::trace()
{
    if(!m_enabled) return ;

    glm::vec3 eye ;
    glm::vec3 U ;
    glm::vec3 V ;
    glm::vec3 W ;

    m_composition->getEyeUVW(eye, U, V, W); // must setModelToWorld in composition first
    //if(m_trace_count == 0) print(eye,U,V,W, "OptiXEngine::trace eye/U/V/W ");

    float scene_epsilon = m_composition->getNear();
    m_context[ "scene_epsilon"]->setFloat(scene_epsilon); 

    m_context[ "eye"]->setFloat( make_float3( eye.x, eye.y, eye.z ) );
    m_context[ "U"  ]->setFloat( make_float3( U.x, U.y, U.z ) );
    m_context[ "V"  ]->setFloat( make_float3( V.x, V.y, V.z ) );
    m_context[ "W"  ]->setFloat( make_float3( W.x, W.y, W.z ) );

    Buffer buffer = m_context["output_buffer"]->getBuffer();
    RTsize buffer_width, buffer_height;
    buffer->getSize( buffer_width, buffer_height );

    unsigned int width  = static_cast<unsigned int>(buffer_width) ;
    unsigned int height = static_cast<unsigned int>(buffer_height) ;

    if(m_trace_count % 100 == 0) LOG(info) << "OptiXEngine::trace " << m_trace_count << " size(" <<  width << "," <<  height << ")";

    m_context->launch( 0,  width, height );

    m_trace_count += 1 ; 
}


void OptiXEngine::initGenerate(NumpyEvt* evt)
{
    NPY* gensteps = evt->getGenstepData();

    assert(gensteps->getDimensions() == 3);
    assert(gensteps->getShape(1) == 6);
    assert(gensteps->getShape(2) == 4);

    int genstep_buffer_id = gensteps->getBufferId();
    if(genstep_buffer_id > -1)
    {
        unsigned int genstep_count = gensteps->getShape(0);
        unsigned int genstep_sizeq  = gensteps->getShape(1);  // in quads
        unsigned int genstep_totquad = genstep_count * genstep_sizeq ;  
        assert(genstep_sizeq == 6);

        m_genstep_buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, genstep_buffer_id);
        m_genstep_buffer->setFormat(RT_FORMAT_FLOAT4);
        m_genstep_buffer->setSize( genstep_totquad );
        m_context["genstep_buffer"]->set( m_genstep_buffer );

        LOG(info) << "OptiXEngine::initGenerate "
                  << " genstep_buffer_id " << genstep_buffer_id 
                  << " genstep_count " << genstep_count 
                  << " genstep_sizeq " << genstep_sizeq 
                  << " genstep_totquad " << genstep_totquad  ;
    } 


    NPY* photons = evt->getPhotonData();

    int photon_buffer_id = photons ? photons->getBufferId() : -1 ;
    if(photon_buffer_id > -1)
    {
        unsigned int photon_count = photons->getShape(0);
        unsigned int photon_sizeq = photons->getShape(1);  
        unsigned int photon_totquad = photon_count * photon_sizeq ;  
        assert(photon_sizeq == 1);

        m_photon_buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, photon_buffer_id);
        m_photon_buffer->setFormat(RT_FORMAT_FLOAT4);
        m_photon_buffer->setSize( photon_totquad );
        m_context["photon_buffer"]->set( m_photon_buffer );

        LOG(info) << "OptiXEngine::initGenerate "
                  << " photon_buffer_id " << photon_buffer_id 
                  << " photon_count " << photon_count 
                  << " photon_sizeq " << photon_sizeq 
                  << " photon_totquad " << photon_totquad  ;
 

    }
}



void OptiXEngine::generate()
{
}





optix::Buffer OptiXEngine::createOutputBuffer(RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;
    buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, format, width, height);
    return buffer ; 
}

optix::Buffer OptiXEngine::createOutputBuffer_VBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;

    glGenBuffers(1, &id);
    glBindBuffer(GL_ARRAY_BUFFER, id);

    size_t element_size ; 
    m_context->checkError(rtuGetSizeForRTformat(format, &element_size));
    assert(element_size == 16);

    const GLvoid *data = NULL ;
    glBufferData(GL_ARRAY_BUFFER, element_size * width * height, data, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0); 

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, id);
    buffer->setFormat(format);
    buffer->setSize( width, height );

    return buffer;
}

optix::Buffer OptiXEngine::createOutputBuffer_PBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;

    glGenBuffers(1, &id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, id);

    size_t element_size ; 
    m_context->checkError(rtuGetSizeForRTformat(format, &element_size));
    assert(element_size == 4);

    unsigned int nbytes = element_size * width * height ;

    m_pbo_data = (unsigned char*)malloc(nbytes);
    memset(m_pbo_data, 0x88, nbytes);  // initialize PBO to grey 

    glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, m_pbo_data, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); 

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, id);
    buffer->setFormat(format);
    buffer->setSize( width, height );

    LOG(info) << "OptiXEngine::createOutputBuffer_PBO  element_size " << element_size << " size (" << width << "," << height << ") pbo id " << id ;
  
    return buffer;
}

void OptiXEngine::associate_PBO_to_Texture(unsigned int texId)
{
    printf("OptiXEngine::associate_PBO_to_Texture texId %u \n", texId);

    assert(m_pbo > 0);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture( GL_TEXTURE_2D, texId );

    // this kills the teapot
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL );
}


void OptiXEngine::push_PBO_to_Texture(unsigned int texId)
{
    //printf("OptiXEngine::push_PBO_to_Texture texId %u \n", texId);
   // see  GLUTDisplay::displayFrame() 

    RTsize buffer_width_rts, buffer_height_rts;
    m_output_buffer->getSize( buffer_width_rts, buffer_height_rts );

    int buffer_width  = static_cast<int>(buffer_width_rts);
    int buffer_height = static_cast<int>(buffer_height_rts);

    RTformat buffer_format = m_output_buffer->getFormat();

    //
    // glTexImage2D specifies mutable texture storage characteristics and provides the data
    //
    //    *internalFormat* 
    //         format with which OpenGL should store the texels in the texture
    //    *data*
    //         location of the initial texel data in host memory, 
    //         if a buffer is bound to the GL_PIXEL_UNPACK_BUFFER binding point, 
    //         texel data is read from that buffer object, and *data* is interpreted 
    //         as an offset into that buffer object from which to read the data. 
    //    *format* and *type*
    //         initial source texel data layout which OpenGL will convert 
    //         to the internalFormat
    // 

   // send pbo to texture

    assert(m_pbo > 0);

    glBindTexture(GL_TEXTURE_2D, texId );

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);

    RTsize elementSize = m_output_buffer->getElementSize();
    if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    switch(buffer_format) 
    {   //               target   miplevl  internalFormat                     border  format   type           data  
        case RT_FORMAT_UNSIGNED_BYTE4:
            //printf("OptiXEngine::push_PBO_to_Texture RT_FORMAT_UNSIGNED_BYTE4 tex:%d \n", texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, buffer_width, buffer_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
            break ; 
        case RT_FORMAT_FLOAT4:
            printf("OptiXEngine::push_PBO_to_Texture RT_FORMAT_FLOAT4 tex:%d\n", texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, buffer_width, buffer_height, 0, GL_RGBA, GL_FLOAT, 0);
            break;
        case RT_FORMAT_FLOAT3:
            printf("OptiXEngine::push_PBO_to_Texture RT_FORMAT_FLOAT3 tex:%d\n", texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, buffer_width, buffer_height, 0, GL_RGB, GL_FLOAT, 0);
            break;
        case RT_FORMAT_FLOAT:
            printf("OptiXEngine::push_PBO_to_Texture RT_FORMAT_FLOAT tex:%d\n", texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, buffer_width, buffer_height, 0, GL_LUMINANCE, GL_FLOAT, 0);
            break;
        default:
            assert(0 && "Unknown buffer format");
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


}


void OptiXEngine::render()
{
    if(!m_enabled) return ;
    push_PBO_to_Texture(m_texture_id);
    m_renderer->render();
}






void OptiXEngine::fill_PBO()
{
    // not working
    //
    //  https://www.opengl.org/wiki/Pixel_Buffer_Object
    //  https://www.opengl.org/wiki/Pixel_Transfer

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    void* pboData = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);

    for(unsigned int w=0 ; w<m_width ; ++w ){
    for(unsigned int h=0 ; h<m_height ; ++h ) 
    {
        unsigned char* p = (unsigned char*)pboData ; 
        *(p+0) = 0xAA ;
        *(p+1) = 0xBB ;
        *(p+2) = 0xCC ;
        *(p+3) = 0x00 ;
    }
    } 
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER); 
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}




void OptiXEngine::cleanUp()
{
    if(!m_enabled) return ;
    saveAccelCache();
    m_context->destroy();
    m_context = 0;
}

optix::Context& OptiXEngine::getContext()
{
    return m_context ; 
}

void OptiXEngine::setSize(unsigned int width, unsigned int height)
{
    m_width = width ;
    m_height = height ;

    m_composition->setSize(width, height);
    m_texture->setSize(width, height);
}

/*
124 void SampleScene::resize(unsigned int width, unsigned int height)
125 {
126   try {
127     Buffer buffer = getOutputBuffer();
128     buffer->setSize( width, height );
129 
130     if(m_use_vbo_buffer)
131     {
132       buffer->unregisterGLBuffer();
133       glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer->getGLBOId());
134       glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
135       glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
136       buffer->registerGLBuffer();
137     }

*/

///usr/local/env/cuda/OptiX_370b2_sdk/sutil/MeshScene.cpp




void OptiXEngine::loadAccelCache()
{
  // If acceleration caching is turned on and a cache file exists, load it.
  if( m_accel_caching_on ) {
  
    if(m_filename.empty()) return ;  

    const std::string cachefile = getCacheFileName();
    LOG(info)<<"OptiXEngine::loadAccelCache cachefile " << cachefile ; 

    std::ifstream in( cachefile.c_str(), std::ifstream::in | std::ifstream::binary );
    if ( in ) {
      unsigned long long int size = 0ull;

#ifdef WIN32
      // This is to work around a bug in Visual Studio where a pos_type is cast to an int before being cast to the requested type, thus wrapping file sizes at 2 GB. WTF? To be fixed in VS2011.
      FILE *fp = fopen(cachefile.c_str(), "rb");

      _fseeki64(fp, 0L, SEEK_END);
      size = _ftelli64(fp);
      fclose(fp);
#else
      // Read data from file
      in.seekg (0, std::ios::end);
      std::ifstream::pos_type szp = in.tellg();
      in.seekg (0, std::ios::beg);
      size = static_cast<unsigned long long int>(szp);
#endif

      std::cerr << "acceleration cache file found: '" << cachefile << "' (" << size << " bytes)\n";
      
      if(sizeof(size_t) <= 4 && size >= 0x80000000ULL) {
        std::cerr << "[WARNING] acceleration cache file too large for 32-bit application.\n";
        m_accel_cache_loaded = false;
        return;
      }

      char* data = new char[static_cast<size_t>(size)];
      in.read( data, static_cast<std::streamsize>(size) );
      
      // Load data into accel
      Acceleration accel = m_geometry_group->getAcceleration();
      try {
        accel->setData( data, static_cast<RTsize>(size) );
        m_accel_cache_loaded = true;

      } catch( optix::Exception& e ) {
        // Setting the acceleration cache failed, but that's not a problem. Since the acceleration
        // is marked dirty and builder and traverser are both set already, it will simply build as usual,
        // without using the cache. So we just warn the user here, but don't do anything else.
        std::cerr << "[WARNING] could not use acceleration cache, reason: " << e.getErrorString() << std::endl;
        m_accel_cache_loaded = false;
      }

      delete[] data;

    } else {
      m_accel_cache_loaded = false;
      std::cerr << "no acceleration cache file found\n";
    }
  }
}

void OptiXEngine::saveAccelCache()
{
  // If accel caching on, marshallize the accel 

  if( m_accel_caching_on && !m_accel_cache_loaded ) {

    if(m_filename.empty()) return ;  

    const std::string cachefile = getCacheFileName();

    // Get data from accel
    Acceleration accel = m_geometry_group->getAcceleration();
    RTsize size = accel->getDataSize();
    char* data  = new char[size];
    accel->getData( data );

    // Write to file
    LOG(info)<<"OptiXEngine::saveAccelCache cachefile " << cachefile << " size " << size ;  

    std::ofstream out( cachefile.c_str(), std::ofstream::out | std::ofstream::binary );
    if( !out ) {
      delete[] data;
      std::cerr << "could not open acceleration cache file '" << cachefile << "'" << std::endl;
      return;
    }
    out.write( data, size );
    delete[] data;
    std::cerr << "acceleration cache written: '" << cachefile << "'" << std::endl;
  }
}

std::string OptiXEngine::getCacheFileName()
{
  std::string cachefile = m_filename;
  size_t idx = cachefile.find_last_of( '.' );
  cachefile.erase( idx );
  cachefile.append( ".accelcache" );
  return cachefile;
}


