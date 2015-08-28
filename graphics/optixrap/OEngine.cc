#include "OEngine.hh"

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
#include <fstream>

// npy-
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NumpyEvt.hpp"

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
//#include "GGeoOptiXGeometry.hh"
#include "OGeo.hh"
#include "OBoundaryLib.hh"

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

    LOG(info) << "OEngine::init " 
              << " mode " << getModeName()
              ; 
    m_context = Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    //m_context->setPrintLaunchIndex(0,0,0);
    m_context->setStackSize( 2180 ); // TODO: make externally configurable, and explore performance implications
    m_config = RayTraceConfig::makeInstance(m_context, m_cmake_target);


    m_top = m_context->createGroup();

    m_domain = NPY<float>::make_vec4(e_number_domain,1,0.f) ;
    m_idomain = NPY<int>::make_vec4(e_number_idomain,1,0) ;

    if(!isCompute())
    {
        initRenderer();
    } 

    initRayTrace();
    initGeometry();

    if(m_evt)
    {
        // TODO: move these elsewhere, only needed on NPY arrival
        initGenerate();  
        initRng();    
    }

    preprocess();  // context is validated and accel structure built in here

    LOG(info) << "OEngine::init DONE " ;
}



void OEngine::initRenderer()
{
    unsigned int width  = m_composition->getPixelWidth();
    unsigned int height = m_composition->getPixelHeight();

    m_renderer = new Renderer("tex");
    m_texture = new Texture();   // QuadTexture would be better name
    m_texture->setSize(width, height);
    m_texture->create();
    m_texture_id = m_texture->getTextureId() ;

    LOG(debug) << "OEngine::initRenderer size(" << width << "," << height << ")  texture_id " << m_texture_id ;
    m_renderer->upload(m_texture);
}


unsigned int OEngine::getNumEntryPoint()
{
    return m_evt ? e_entryPointCount : e_entryPointCount - 1 ; 
}

void OEngine::initRayTrace()
{
    if(isInterop())
    {
        unsigned int width  = m_composition->getPixelWidth();
        unsigned int height = m_composition->getPixelHeight();
        LOG(debug) << "OEngine::initContext size (" << width << "," << height << ")" ;
        m_output_buffer = createOutputBuffer_PBO(m_pbo, RT_FORMAT_UNSIGNED_BYTE4, width, height) ;
        m_touch_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT4, 1, 1);
    } 
    else
    {
        m_output_buffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, 0);
        m_touch_buffer  = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT4, 1, 1);
    } 
    m_context["output_buffer"]->set( m_output_buffer );
    m_context["touch_buffer"]->set( m_touch_buffer );
    m_context["touch_mode" ]->setUint( 0u );


    // "touch" mode is tied to the active rendering (currently only e_pinhole_camera)
    // as the meaning of x,y mouse/trackpad touches depends on that rendering.  
    // Because of this using a separate "touch" entry point may not so useful ?
    // Try instead splitting at ray type level.
    //
    // But the output requirements are very different ? Which would argue for a separate entry point.
    //

    unsigned int numEntryPoint = getNumEntryPoint();
    m_context->setEntryPointCount( numEntryPoint );  

    m_config->setRayGenerationProgram(  e_pinhole_camera, "pinhole_camera.cu", "pinhole_camera" );
    m_config->setExceptionProgram(      e_pinhole_camera, "pinhole_camera.cu", "exception");

    m_context[ "radiance_ray_type"   ]->setUint( e_radiance_ray );
    m_context[ "touch_ray_type"      ]->setUint( e_touch_ray );
    m_context[ "propagate_ray_type"  ]->setUint( e_propagate_ray );
    m_context[ "bounce_max"          ]->setUint( m_bounce_max );
    m_context[ "record_max"          ]->setUint( m_record_max );

    m_context->setRayTypeCount( e_rayTypeCount );

    m_config->setMissProgram( e_radiance_ray , "constantbg.cu", "miss" );

    m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f ); // map(int,np.array([0.34,0.55,0.85])*255) -> [86, 140, 216]
    m_context[ "bad_color" ]->setFloat( 1.0f, 0.0f, 0.0f );
}






// fulfil Touchable interface
unsigned int OEngine::touch(int ix_, int iy_)
{

    if(m_trace_count == 0)
    {
        LOG(warning) << "OEngine::touch \"OptiX touch mode\" only works after performing an OptiX trace, press O to toggle OptiX tracing then try again " ; 
        return 0 ; 
    }


    // (ix_, iy_) 
    //        (0,0)              at top left,  
    //   (1024,768)*pixel_factor at bottom right


    RTsize width, height;
    m_output_buffer->getSize( width, height );

    int ix = ix_ ; 
    int iy = height - iy_;   

    // (ix,iy) 
    //   (0,0)                     at bottom left
    //   (1024,768)*pixel_factor   at top right  

    m_context["touch_mode"]->setUint(1u);
    m_context["touch_index"]->setUint(ix, iy ); // by inspection
    m_context["touch_dim"]->setUint(width, height);

    RTsize touch_width = 1u ; 
    RTsize touch_height = 1u ; 

    // TODO: generalize touch to work with the active camera (eg could be orthographic)
    m_context->launch( e_pinhole_camera , touch_width, touch_height );

    Buffer touchBuffer = m_context[ "touch_buffer"]->getBuffer();
    m_context["touch_mode"]->setUint(0u);

    uint4* touchBuffer_Host = static_cast<uint4*>( touchBuffer->map() );
    uint4 touch = touchBuffer_Host[0] ;
    touchBuffer->unmap();

    LOG(info) << "OEngine::touch "
              << " ix_ " << ix_ 
              << " iy_ " << iy_   
              << " ix " << ix 
              << " iy " << iy   
              << " width " << width   
              << " height " << height 
              << " touch.x nodeIndex " << touch.x 
              << " touch.y " << touch.y 
              << " touch.z " << touch.z   
              << " touch.w " << touch.w 
              ;  

     unsigned int target = touch.x ; 
    // seems out of place
    // m_composition->setTarget(target); 

    return target ; 
}




void OEngine::initGeometry()
{
    LOG(info) << "OEngine::initGeometry" ;

    GGeo* gg = getGGeo();
    assert(gg) ; 

    GBoundaryLib* glib = getBoundaryLib();
    assert(glib);

    OBoundaryLib olib(m_context, glib);
    olib.convert(); 

    OGeo         og(m_context, gg);
    og.setTop(m_top);
    og.convert(); 
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
    m_context[ "propagate_epsilon"]->setFloat(pe); 
 
    LOG(info)<< "OEngine::preprocess start validate ";
    m_context->validate();
    LOG(info)<< "OEngine::preprocess start compile ";
    m_context->compile();
    LOG(info)<< "OEngine::preprocess start building Accel structure ";
    m_context->launch(e_pinhole_camera,0); 
    //m_context->launch(e_generate,0); 

    LOG(info)<< "OEngine::preprocess DONE ";
}

void OEngine::trace()
{
    if(!m_enabled) return ;


    glm::vec3 eye ;
    glm::vec3 U ;
    glm::vec3 V ;
    glm::vec3 W ;

    m_composition->getEyeUVW(eye, U, V, W); // must setModelToWorld in composition first
    //if(m_trace_count == 0) print(eye,U,V,W, "OEngine::trace eye/U/V/W ");

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

    if(m_trace_count % 100 == 0) LOG(info) << "OEngine::trace " << m_trace_count << " size(" <<  width << "," <<  height << ")";

    m_context->launch( e_pinhole_camera,  width, height );

    m_trace_count += 1 ; 
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

    m_context->launch( e_generate,  width, height );

    m_generate_count += 1 ; 
}


void OEngine::initGenerate()
{
    if(!m_evt) return ;
    initGenerate(m_evt);
}

void OEngine::initGenerate(NumpyEvt* evt)
{
    const char* raygenprg = m_trivial ? "trivial" : "generate" ; 

    LOG(info) << "OEngine::initGenerate " << raygenprg ; 
    m_config->setRayGenerationProgram(e_generate, "generate.cu", raygenprg );
    m_config->setExceptionProgram(    e_generate, "generate.cu", "exception");

    NPY<float>* gensteps =  evt->getGenstepData() ;

    m_genstep_buffer = createIOBuffer<float>( gensteps );
    m_context["genstep_buffer"]->set( m_genstep_buffer );

    if(isCompute())
    {
        LOG(info) << "OEngine::initGenerate (COMPUTE)" 
                  << " uploading gensteps "
                  ;
        upload(m_genstep_buffer, gensteps);
    }

    m_photon_buffer = createIOBuffer<float>( evt->getPhotonData() );
    m_context["photon_buffer"]->set( m_photon_buffer );

    m_record_buffer = createIOBuffer<short>( evt->getRecordData() );
    m_context["record_buffer"]->set( m_record_buffer );

    m_sequence_buffer = createIOBuffer<unsigned long long>( evt->getSequenceData() );
    m_context["sequence_buffer"]->set( m_sequence_buffer );

    m_phosel_buffer = createIOBuffer<unsigned char>( evt->getPhoselData() );
    m_context["phosel_buffer"]->set( m_phosel_buffer );

    m_recsel_buffer = createIOBuffer<unsigned char>( evt->getRecselData() );
    if(m_recsel_buffer.get())   
        m_context["recsel_buffer"]->set( m_recsel_buffer );

    // need to have done scene.uploadSelection for the recsel to have a buffer_id
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
optix::Buffer OEngine::createIOBuffer(NPY<T>* npy)
{
    assert(npy);
    unsigned int ni = npy->getShape(0);
    unsigned int nj = npy->getShape(1);  
    unsigned int nk = npy->getShape(2);  

    LOG(info)<<"OEngine::createIOBuffer "
             << " ni " << ni 
             << " nj " << nj 
             << " nk " << nk  
             ;

    Buffer buffer;
    if(isInterop())
    {
        int buffer_id = npy ? npy->getBufferId() : -1 ;
        if(buffer_id > -1 )
        {
            buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, buffer_id);
        } 
        else
        {
            LOG(warning) << "OEngine::createIOBuffer CANNOT createBufferFromGLBO as not uploaded  "
                         << " buffer_id " << buffer_id 
                         ; 
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
        LOG(info) << "OEngine::createIOBuffer "
                  << " RT_FORMAT_USER " 
                  << " elementsize " << sizeof(T)
                  << " size " << size 
                  ;
    }
    else
    {
        size = ni*nj ; 
        LOG(info) << "OEngine::createIOBuffer "
                  << " (quad) " 
                  << " size " << size 
                  ;

    }

    buffer->setSize(size);
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







#if 0
CUdeviceptr OEngine::getHistoryBufferDevicePointer(unsigned int optix_device_number)
{

/*
// CUdeviceptr is typedef to unsigned long long 
 /Developer/OptiX/include/optix_cuda_interop.h 

122   * @ref rtBufferGetDevicePointer returns the pointer to the data of \a buffer on device \a optix_device_number in **\a device_pointer.
123   *
124   * If @ref rtBufferGetDevicePointer has been called for a single device for a given buffer,
125   * the user can change the buffer's content on that device. OptiX must then synchronize the new buffer contents to all devices.
126   * These synchronization copies occur at every @ref rtContextLaunch "rtContextLaunch", unless the buffer is declared with @ref RT_BUFFER_COPY_ON_DIRTY.
127   * In this case, use @ref rtBufferMarkDirty to notify OptiX that the buffer has been dirtied and must be synchronized.
128   *


   Sequence of actions...

        * create OpenGL buffers in Scene::uploadEvt, with OpenGL buffer_id tucked away in the NPY
        * OEngine::initGenerate takes hold of the buffers via their buffer_id and 
          places references to them in the OptiX context
        * OEngine::generate populates buffers during OptiX context launches  

        * post generate 
 
*/
}
#endif


optix::Buffer OEngine::createOutputBuffer(RTformat format, unsigned int width, unsigned int height)
{
    Buffer buffer;
    buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, format, width, height);
    return buffer ; 
}

optix::Buffer OEngine::createOutputBuffer_VBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height)
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

optix::Buffer OEngine::createOutputBuffer_PBO(unsigned int& id, RTformat format, unsigned int width, unsigned int height)
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

    LOG(info) << "OEngine::createOutputBuffer_PBO  element_size " << element_size << " size (" << width << "," << height << ") pbo id " << id ;
  
    return buffer;
}

void OEngine::associate_PBO_to_Texture(unsigned int texId)
{
    printf("OEngine::associate_PBO_to_Texture texId %u \n", texId);

    assert(m_pbo > 0);
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture( GL_TEXTURE_2D, texId );

    // this kills the teapot
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL );
}


void OEngine::push_PBO_to_Texture(unsigned int texId)
{
    //printf("OEngine::push_PBO_to_Texture texId %u \n", texId);
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
            //printf("OEngine::push_PBO_to_Texture RT_FORMAT_UNSIGNED_BYTE4 tex:%d \n", texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, buffer_width, buffer_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
            break ; 
        case RT_FORMAT_FLOAT4:
            printf("OEngine::push_PBO_to_Texture RT_FORMAT_FLOAT4 tex:%d\n", texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, buffer_width, buffer_height, 0, GL_RGBA, GL_FLOAT, 0);
            break;
        case RT_FORMAT_FLOAT3:
            printf("OEngine::push_PBO_to_Texture RT_FORMAT_FLOAT3 tex:%d\n", texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, buffer_width, buffer_height, 0, GL_RGB, GL_FLOAT, 0);
            break;
        case RT_FORMAT_FLOAT:
            printf("OEngine::push_PBO_to_Texture RT_FORMAT_FLOAT tex:%d\n", texId);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, buffer_width, buffer_height, 0, GL_LUMINANCE, GL_FLOAT, 0);
            break;
        default:
            assert(0 && "Unknown buffer format");
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    //glBindTexture(GL_TEXTURE_2D, 0 );   get blank screen when do this here

}


void OEngine::render()
{
    if(!m_enabled) return ;

    push_PBO_to_Texture(m_texture_id);
    m_renderer->render();

    glBindTexture(GL_TEXTURE_2D, 0 );  
}


void OEngine::fill_PBO()
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




void OEngine::cleanUp()
{
    if(!m_enabled) return ;
    saveAccelCache();
    m_context->destroy();
    m_context = 0;
}

optix::Context& OEngine::getContext()
{
    return m_context ; 
}

void OEngine::setSize(unsigned int width, unsigned int height)
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


