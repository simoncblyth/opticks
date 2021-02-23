/**
UseOptiX7GeometryInstancedGASCompDyn
======================================

**/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include "Util.h"
#include "Ctx.h"
#include "Params.h"
#include "Frame.h"
#include "Geo.h"
#include "PIP.h"
#include "SBT.h"

#include <glm/glm.hpp>

int main(int argc, char** argv)
{
    const char* spec = argc > 1 ? argv[1] : "i0" ; 
    std::cout << argv[0] << " spec " << spec << std::endl ;  

    const char* name = "UseOptiX7GeometryInstancedGASCompDyn" ; 
    const char* prefix = getenv("PREFIX");  assert( prefix && "expecting PREFIX envvar pointing to writable directory" );

    const char* cmake_target = name ; 
    const char* ptx_path = Util::PTXPath( prefix, cmake_target, name ) ; 
    const char* ppm_path = Util::PPMPath( prefix, name ); 
    std::cout << " ptx_path " << ptx_path << std::endl ; 

    unsigned width = 1024u ; 
    unsigned height = 768u ; 
    unsigned depth = 1u ; 

    Ctx ctx ; 
    Geo geo(spec);   // must be after Ctx creation as creates GAS

    float top_extent = geo.getTopExtent() ;  
    glm::vec4 ce(0.f,0.f,0.f, top_extent*1.4f );   // defines the center-extent of the region to view
    glm::vec3 eye,U,V,W  ;
    Util::GetEyeUVW( ce, width, height, eye, U, V, W ); 

    ctx.setView(eye, U, V, W, geo.tmin, geo.tmax ); 
    ctx.setSize(width, height, depth); 

    PIP pip(ptx_path); 
    SBT sbt(&pip, ctx.params);
    sbt.setGeo(&geo); 

    Frame frame(ctx.params); 

    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    ctx.uploadParams();  
    OPTIX_CHECK( optixLaunch( pip.pipeline, stream, ctx.d_param, sizeof( Params ), &(sbt.sbt), ctx.params->width, ctx.params->height, ctx.params->depth ) );
    CUDA_SYNC_CHECK();

    frame.download(); 
    frame.writePPM(ppm_path);  

    return 0 ; 
}
