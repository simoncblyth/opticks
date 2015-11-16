#include "OTracer.hh"
#include "OContext.hh"
#include "OTimes.hh"

#include <iomanip>


#include <optixu/optixu.h>
#include <optixu/optixu_math_stream_namespace.h>

// npy-
#include "GLMPrint.hpp"
#include "timeutil.hpp"

// oglrap-
#include "Composition.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

using namespace optix ; 


void OTracer::init()
{
    m_context = m_ocontext->getContext();

    m_ocontext->setRayGenerationProgram(  OContext::e_pinhole_camera_entry, "pinhole_camera.cu.ptx", "pinhole_camera" );
    m_ocontext->setExceptionProgram(      OContext::e_pinhole_camera_entry, "pinhole_camera.cu.ptx", "exception");
    m_ocontext->setMissProgram(           OContext::e_radiance_ray , "constantbg.cu.ptx", "miss" );

    m_context[ "scene_epsilon"]->setFloat(m_composition->getNear());

    m_context[ "radiance_ray_type"   ]->setUint( OContext::e_radiance_ray );
    m_context[ "touch_ray_type"      ]->setUint( OContext::e_touch_ray );
    m_context[ "propagate_ray_type"  ]->setUint( OContext::e_propagate_ray );


    m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f ); // map(int,np.array([0.34,0.55,0.85])*255) -> [86, 140, 216]
    m_context[ "bad_color" ]->setFloat( 1.0f, 0.0f, 0.0f );

    m_trace_times = new OTimes ; 
}


void OTracer::trace()
{
    LOG(debug) << "OTracer::trace " << m_trace_count ; 

    double t0 = getRealTime();

    glm::vec3 eye ;
    glm::vec3 U ;
    glm::vec3 V ;
    glm::vec3 W ;

    m_composition->getEyeUVW(eye, U, V, W); // must setModelToWorld in composition first

    bool parallel = m_composition->getParallel();
    float scene_epsilon = m_composition->getNear();

    m_context[ "parallel"]->setUint( parallel ? 1u : 0u); 
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
         LOG(info) << "OTracer::trace " 
                   << " trace_count " << m_trace_count 
                   << " resolution_scale " << m_resolution_scale 
                   << " size(" <<  width << "," <<  height << ")";


    double t1 = getRealTime();

    m_ocontext->launch( OContext::e_pinhole_camera_entry,  width, height, m_trace_times );

    double t2 = getRealTime();

    m_trace_count += 1 ; 
    m_trace_prep += t1 - t0 ; 
    m_trace_time += t2 - t1 ; 

    LOG(debug) << m_trace_times->description("OTracer::trace m_trace_times") ;


}



void OTracer::report(const char* msg)
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



