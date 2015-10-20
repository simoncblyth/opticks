#include "OEngine.hh"
#include "OContext.hh"
#include "OFrame.hh"

#include "assert.h"
#include "stdio.h"
#include "string.h"

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
#include <iomanip>
#include <fstream>

// npy-
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NumpyEvt.hpp"
#include "Timer.hpp"
#include "timeutil.hpp"

// oglrap-
#include "Composition.hh"

#include "Renderer.hh"
#include "Texture.hh"

#include "Rdr.hh"

// ggeo-
#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GBoundaryLib.hh"

// optixrap-
#include "RayTraceConfig.hh"
#include "OGeo.hh"
#include "OBoundaryLib.hh"
#include "OBuf.hh"
#include "OBufPair.hh"

// cudawrap-

using namespace optix ; 
#include "cuRANDWrapper.hh"
#include "curand.h"
#include "curand_kernel.h"

// extracts from /usr/local/env/cuda/OptiX_370b2_sdk/sutil/SampleScene.cpp

const char* OEngine::COMPUTE_ = "COMPUTE" ; 
const char* OEngine::INTEROP_ = "INTEROP" ; 

const char* OEngine::getModeName()
{
    switch(m_mode)
    {
       case COMPUTE:return COMPUTE_ ; break ; 
       case INTEROP:return INTEROP_ ; break ; 
    }
    assert(0);
}

// interop and compute are sufficiently different to warrant a separate class ?
// maybe with common base class


void OEngine::init()
{
    if(!m_enabled) return ;

    m_timer      = new Timer("OEngine");
    m_timer->setVerbose(true);
   // m_timer->start();

    LOG(info) << "OEngine::init " 
              << " mode " << getModeName()
              ; 


    optix::uint4 debugControl = optix::make_uint4(m_debug_photon,0,0,0);
    LOG(info) << "OEngine::init debugControl " 
              << " x " << debugControl.x 
              << " y " << debugControl.y
              << " z " << debugControl.z 
              << " w " << debugControl.w 
              ;

    m_context["debug_control"]->setUint(debugControl); 


    // fallbacks are these seem not to being set 
    m_context["instance_index"]->setUint( 0u ); 
    m_context["primitive_count"]->setUint( 0u );


    m_config = RayTraceConfig::makeInstance(m_context);


    m_domain = NPY<float>::make(e_number_domain,1,4) ;
    m_domain->fill(0.f);
    m_idomain = NPY<int>::make(e_number_idomain,1,4) ;
    m_idomain->fill(0);


    initRayTrace();
    initGeometry();

    if(m_evt)
    {
        // TODO: move these elsewhere, only needed on NPY arrival
        //       split once onlys from on arrival of evt 
        //
        initGenerateOnce();  
        initRng();    

        initGenerate();  
    }

    preprocess();  // context is validated and accel structure built in here

    LOG(info) << "OEngine::init DONE " ;
}






void OEngine::initRayTrace()
{
    m_config->setRayGenerationProgram(  OContext::e_pinhole_camera_entry, "pinhole_camera.cu.ptx", "pinhole_camera" );
    m_config->setExceptionProgram(      OContext::e_pinhole_camera_entry, "pinhole_camera.cu.ptx", "exception");

    m_context[ "radiance_ray_type"   ]->setUint( OContext::e_radiance_ray );
    m_context[ "touch_ray_type"      ]->setUint( OContext::e_touch_ray );
    m_context[ "propagate_ray_type"  ]->setUint( OContext::e_propagate_ray );

    m_context[ "bounce_max"          ]->setUint( m_bounce_max );
    m_context[ "record_max"          ]->setUint( m_record_max );

    m_config->setMissProgram(  OContext::e_radiance_ray , "constantbg.cu.ptx", "miss" );

    m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f ); // map(int,np.array([0.34,0.55,0.85])*255) -> [86, 140, 216]
    m_context[ "bad_color" ]->setFloat( 1.0f, 0.0f, 0.0f );
}



void OEngine::initGeometry()
{
    LOG(info) << "OEngine::initGeometry" ;

    LOG(info) << "OEngine::initGeometry calling setupAcceleration" ; 

    loadAccelCache();

    m_context[ "top_object" ]->set( m_top );


    glm::vec4 ce = m_composition->getDomainCenterExtent();
    glm::vec4 td = m_composition->getTimeDomain();

    m_context["center_extent"]->setFloat( make_float4( ce.x, ce.y, ce.z, ce.w ));
    m_context["time_domain"]->setFloat(   make_float4( td.x, td.y, td.z, td.w ));

    m_domain->setQuad(e_center_extent, 0, ce );
    m_domain->setQuad(e_time_domain  , 0, td );

    glm::vec4 wd ;
    m_context["wavelength_domain"]->getFloat(wd.x,wd.y,wd.z,wd.w);
    m_domain->setQuad(e_wavelength_domain  , 0, wd );

    print(wd, "OEngine::initGeometry wavelength_domain");

    glm::ivec4 ci ;
    ci.x = m_bounce_max ;  
    ci.y = m_rng_max ;  
    ci.z = 0 ;  
    ci.w = m_evt ? m_evt->getMaxRec() : 0 ;  

    m_idomain->setQuad(e_config_idomain, 0, ci );

    // cf with MeshViewer::initGeometry
    LOG(info) << "OEngine::initGeometry DONE "
              << " x: " << ce.x  
              << " y: " << ce.y  
              << " z: " << ce.z  
              << " w: " << ce.w  ;
}


void OEngine::preprocess()
{
    LOG(info)<< "OEngine::preprocess";

    m_context[ "scene_epsilon"]->setFloat(m_composition->getNear());
    float pe = 0.1f ; 
    m_context[ "propagate_epsilon"]->setFloat(pe);  // TODO: check impact of changing propagate_epsilon
 

    // repeating for the hell of it,  
    // shows that even 0-work launches are taking significant time every time
    
    unsigned int n = 1;  
    for(unsigned int i=0 ; i < n ; i++)
    { 
        m_ocontext->launch( OContext::e_pinhole_camera_entry, 0, 0, m_prep_times ); 
        LOG(info) << m_prep_times->description("OEngine::preprocess") ;
    }

}


void OEngine::trace()
{
    if(!m_enabled) return ;

    LOG(info) << "OEngine::trace " << m_trace_count ; 

    double t0 = getRealTime();

    glm::vec3 eye ;
    glm::vec3 U ;
    glm::vec3 V ;
    glm::vec3 W ;

    m_composition->getEyeUVW(eye, U, V, W); // must setModelToWorld in composition first

    float scene_epsilon = m_composition->getNear();

    m_context[ "scene_epsilon"]->setFloat(scene_epsilon); 
    m_context[ "eye"]->setFloat( make_float3( eye.x, eye.y, eye.z ) );
    m_context[ "U"  ]->setFloat( make_float3( U.x, U.y, U.z ) );
    m_context[ "V"  ]->setFloat( make_float3( V.x, V.y, V.z ) );
    m_context[ "W"  ]->setFloat( make_float3( W.x, W.y, W.z ) );

    Buffer buffer = m_context["output_buffer"]->getBuffer();
    RTsize buffer_width, buffer_height;
    buffer->getSize( buffer_width, buffer_height );

    // resolution_scale 
    //
    //   1: full resolution, launch index for every pixel 
    //   2: half resolution, each launch index result duplicated into 2*2=4 pixels
    //            
    unsigned int width  = static_cast<unsigned int>(buffer_width)/m_resolution_scale ;
    unsigned int height = static_cast<unsigned int>(buffer_height)/m_resolution_scale ;
    m_context["resolution_scale"]->setUint( m_resolution_scale ) ;  

    if(m_trace_count % 100 == 0) 
         LOG(info) << "OEngine::trace " 
                   << " trace_count " << m_trace_count 
                   << " resolution_scale " << m_resolution_scale 
                   << " size(" <<  width << "," <<  height << ")";


    double t1 = getRealTime();

    m_ocontext->launch( OContext::e_pinhole_camera_entry,  width, height, m_trace_times );

    double t2 = getRealTime();

    m_trace_count += 1 ; 
    m_trace_prep += t1 - t0 ; 
    m_trace_time += t2 - t1 ; 

    LOG(info) << m_trace_times->description("OEngine::trace m_trace_times") ;


}

void OEngine::initRng()
{
    if(m_rng_max == 0 ) return ;

    const char* rngCacheDir = RayTraceConfig::RngDir() ;
    m_rng_wrapper = cuRANDWrapper::instanciate( m_rng_max, rngCacheDir );

    // OptiX owned RNG states buffer (not CUDA owned)
    m_rng_states = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER);
    m_rng_states->setElementSize(sizeof(curandState));
    m_rng_states->setSize(m_rng_max);
    m_context["rng_states"]->setBuffer(m_rng_states);

    curandState* host_rng_states = static_cast<curandState*>( m_rng_states->map() );
    m_rng_wrapper->setItems(m_rng_max);
    m_rng_wrapper->fillHostBuffer(host_rng_states, m_rng_max);
    m_rng_states->unmap();

    //
    // TODO: investigate Thrust based alternatives to curand initialization 
    //       potential for eliminating cudawrap- 
    //

}


void OEngine::initGenerateOnce()
{
    const char* raygenprg = m_trivial ? "trivial" : "generate" ;  // last ditch debug technique

    LOG(info) << "OEngine::initGenerateOnce " << raygenprg ; 

    m_config->setRayGenerationProgram( OContext::e_generate_entry, "generate.cu.ptx", raygenprg );

    m_config->setExceptionProgram(    OContext::e_generate_entry, "generate.cu.ptx", "exception");
}

void OEngine::initGenerate()
{
    if(!m_evt) return ;
    initGenerate(m_evt);
}

void OEngine::initGenerate(NumpyEvt* evt)
{
    // when isInterop() == true 
    // the OptiX buffers for the evt data are actually references 
    // to the OpenGL buffers created with createBufferFromGLBO
    // by Scene::uploadEvt Scene::uploadSelection

    NPY<float>* gensteps =  evt->getGenstepData() ;

    m_genstep_buffer = createIOBuffer<float>( gensteps, "gensteps");
    m_context["genstep_buffer"]->set( m_genstep_buffer );

    if(isCompute()) 
    {
        LOG(info) << "OEngine::initGenerate (COMPUTE)" 
                  << " uploading gensteps "
                  ;
        upload(m_genstep_buffer, gensteps);
    }
    else if(isInterop())
    {
        assert(gensteps->getBufferId() > 0); 
        LOG(info) << "OEngine::initGenerate (INTEROP)" 
                  << " gensteps handed to OptiX by referencing OpenGL buffer id  "
                  ;
    }


    m_photon_buffer = createIOBuffer<float>( evt->getPhotonData(), "photon" );
    m_context["photon_buffer"]->set( m_photon_buffer );

    m_record_buffer = createIOBuffer<short>( evt->getRecordData(), "record");
    m_context["record_buffer"]->set( m_record_buffer );

    m_sequence_buffer = createIOBuffer<unsigned long long>( evt->getSequenceData(), "sequence" );
    m_context["sequence_buffer"]->set( m_sequence_buffer );

    m_aux_buffer = createIOBuffer<short>( evt->getAuxData(), "aux" );
    if(m_aux_buffer.get())
        m_context["aux_buffer"]->set( m_aux_buffer );

    m_sequence_buf = new OBuf("sequence", m_sequence_buffer );
    m_sequence_buf->setMultiplicity(1u);
    m_sequence_buf->setHexDump(true);

    m_photon_buf = new OBuf("photon", m_photon_buffer );



    // need to have done scene.uploadSelection for the recsel to have a buffer_id
}




void OEngine::generate()
{
    if(!m_enabled) return ;
    if(!m_evt) return ;

    unsigned int numPhotons = m_evt->getNumPhotons();
    assert( numPhotons < getRngMax() && "Use ggeoview-rng-prep to prepare RNG states up to the maximal number of photons generated " );

    unsigned int width  = numPhotons ;
    unsigned int height = 1 ;

    LOG(info) << "OEngine::generate count " << m_generate_count << " size(" <<  width << "," <<  height << ")";

    if(m_override > 0)
    {
        width = m_override ; 
        LOG(warning) << "OEngine::generate OVERRIDE photon count for debugging to " << width ; 
    }

    m_context->launch( OContext::e_generate_entry,  width, height );

    m_generate_count += 1 ; 
}


void OEngine::downloadEvt()
{
    if(!m_evt) return ;
    LOG(info)<<"OEngine::downloadEvt" ;
 
    NPY<float>* dpho = m_evt->getPhotonData();
    download<float>( m_photon_buffer, dpho );

    NPY<short>* drec = m_evt->getRecordData();
    download<short>( m_record_buffer, drec );

    NPY<unsigned long long>* dhis = m_evt->getSequenceData();
    download<unsigned long long>( m_sequence_buffer, dhis );

    LOG(info)<<"OEngine::downloadEvt DONE" ;
}


template <typename T>
void OEngine::upload(optix::Buffer& buffer, NPY<T>* npy)
{
    unsigned int numBytes = npy->getNumBytes(0) ;

    LOG(info)<<"OEngine::upload" 
             << " numBytes " << numBytes 
             ;

    memcpy( buffer->map(), npy->getBytes(), numBytes );
    buffer->unmap(); 
}


template <typename T>
void OEngine::download(optix::Buffer& buffer, NPY<T>* npy)
{
    unsigned int numBytes = npy->getNumBytes(0) ;
    LOG(info)<<"OEngine::download" 
             << " numBytes " << numBytes 
             ;

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 
}


template <typename T>
optix::Buffer OEngine::createIOBuffer(NPY<T>* npy, const char* name)
{
    assert(npy);
    unsigned int ni = npy->getShape(0);
    unsigned int nj = npy->getShape(1);  
    unsigned int nk = npy->getShape(2);  

    Buffer buffer;
    if(isInterop())
    {
        int buffer_id = npy ? npy->getBufferId() : -1 ;
        if(buffer_id > -1 )
        {
            buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, buffer_id);
            LOG(debug) << "OEngine::createIOBuffer (INTEROP) createBufferFromGLBO " 
                      << " name " << std::setw(20) << name
                      << " buffer_id " << buffer_id 
                      << " ( " << ni << "," << nj << "," << nk << ")"
                      ;
        } 
        else
        {
            LOG(warning) << "OEngine::createIOBuffer CANNOT createBufferFromGLBO as not uploaded  "
                         << " name " << std::setw(20) << name
                         << " buffer_id " << buffer_id 
                         ; 

            //assert(0);   only recsel buffer is not uploaded, as kludge interop workaround 
            return buffer ; 
        }
    } 
    else if (isCompute())
    {
        LOG(info) << "OEngine::createIOBuffer (COMPUTE)" ;
        buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
    }


    RTformat format = getFormat(npy->getType());
    buffer->setFormat(format);  // must set format, before can set ElementSize

    unsigned int size ; 
    if(format == RT_FORMAT_USER)
    {
        buffer->setElementSize(sizeof(T));
        size = ni*nj*nk ; 
        LOG(debug) << "OEngine::createIOBuffer "
                  << " RT_FORMAT_USER " 
                  << " elementsize " << sizeof(T)
                  << " size " << size 
                  ;
    }
    else
    {
        size = ni*nj ; 
        LOG(debug) << "OEngine::createIOBuffer "
                  << " (quad) " 
                  << " size " << size 
                  ;

    }

    buffer->setSize(size); // TODO: check without thus, maybe unwise when already referencing OpenGL buffer of defined size
    return buffer ; 
}




RTformat OEngine::getFormat(NPYBase::Type_t type)
{
    RTformat format ; 
    switch(type)
    {
        case NPYBase::FLOAT:     format = RT_FORMAT_FLOAT4         ; break ; 
        case NPYBase::SHORT:     format = RT_FORMAT_SHORT4         ; break ; 
        case NPYBase::INT:       format = RT_FORMAT_INT4           ; break ; 
        case NPYBase::UINT:      format = RT_FORMAT_UNSIGNED_INT4  ; break ; 
        case NPYBase::CHAR:      format = RT_FORMAT_BYTE4          ; break ; 
        case NPYBase::UCHAR:     format = RT_FORMAT_UNSIGNED_BYTE4 ; break ; 
        case NPYBase::ULONGLONG: format = RT_FORMAT_USER           ; break ; 
        case NPYBase::DOUBLE:    format = RT_FORMAT_USER           ; break ; 
    }
    return format ; 
}





void OEngine::report(const char* msg)
{
    LOG(info)<< msg ; 
    if(m_trace_count == 0 ) return ; 

    std::cout 
          << " trace_count     " << std::setw(10) << m_trace_count  
          << " trace_prep      " << std::setw(10) << m_trace_prep   << " avg " << std::setw(10) << m_trace_prep/m_trace_count  << std::endl
          << " trace_time      " << std::setw(10) << m_trace_time   << " avg " << std::setw(10) << m_trace_time/m_trace_count  << std::endl
          << std::endl 
           ;
}




void OEngine::cleanUp()
{
    if(!m_enabled) return ;
    saveAccelCache();
    m_context->destroy();
    m_context = 0;
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




void OEngine::loadAccelCache()
{
  // If acceleration caching is turned on and a cache file exists, load it.
  if( m_accel_caching_on ) {
  
    if(m_filename.empty()) return ;  

    const std::string cachefile = getCacheFileName();
    LOG(info)<<"OEngine::loadAccelCache cachefile " << cachefile ; 

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
      Acceleration accel = m_top->getAcceleration();
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

void OEngine::saveAccelCache()
{
  // If accel caching on, marshallize the accel 

  if( m_accel_caching_on && !m_accel_cache_loaded ) {

    if(m_filename.empty()) return ;  

    const std::string cachefile = getCacheFileName();

    // Get data from accel
    Acceleration accel = m_top->getAcceleration();
    RTsize size = accel->getDataSize();
    char* data  = new char[size];
    accel->getData( data );

    // Write to file
    LOG(info)<<"OEngine::saveAccelCache cachefile " << cachefile << " size " << size ;  

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

std::string OEngine::getCacheFileName()
{
    std::string cachefile = m_filename;
    size_t idx = cachefile.find_last_of( '.' );
    cachefile.erase( idx );
    cachefile.append( ".accelcache" );
    return cachefile;
}


