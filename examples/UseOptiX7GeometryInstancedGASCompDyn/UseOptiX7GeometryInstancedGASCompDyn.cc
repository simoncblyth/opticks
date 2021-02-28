/**
UseOptiX7GeometryInstancedGASCompDyn
======================================

**/
#include <iostream>
#include <cstdlib>
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include "Util.h"
#include "Ctx.h"
#include "Params.h"
#include "Frame.h"
#include "Geo.h"
#include "PIP.h"
#include "SBT.h"

struct AS ; 

int main(int argc, char** argv)
{
    const char* name = "UseOptiX7GeometryInstancedGASCompDyn" ; 
    const char* prefix = getenv("PREFIX");  assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    const char* outdir = getenv("OUTDIR");  assert( outdir && "expecting OUTDIR envvar " );

    const char* cmake_target = name ; 
    const char* ptx_path = Util::PTXPath( prefix, cmake_target, name ) ; 
    std::cout << " ptx_path " << ptx_path << std::endl ; 

    bool small = false ;  
    unsigned width = small ? 512u : 1024u ; 
    unsigned height = small ? 384u : 768u ; 
    unsigned depth = 1u ; 
    unsigned cameratype = Util::GetEValue<unsigned>("CAMERATYPE", 0u ); 

    Geo geo ;  
    geo.write(outdir);  


    // TODO: extracate the view and other params from OptiX/CUDA dependencies
    glm::vec3 eye_model ; 
    Util::GetEVec(eye_model, "EYE", "-1.0,-1.0,1.0"); 

    float top_extent = geo.getTopExtent() ;  
    glm::vec4 ce(0.f,0.f,0.f, top_extent*1.4f );   // defines the center-extent of the region to view
    glm::vec3 eye,U,V,W  ;
    Util::GetEyeUVW( eye_model, ce, width, height, eye, U, V, W ); 

    Ctx ctx ; 
    ctx.setView(eye, U, V, W, geo.tmin, geo.tmax, cameratype ); 
    ctx.setSize(width, height, depth); 



    PIP pip(ptx_path); 
    SBT sbt(&pip);
    sbt.setGeo(&geo); 

    AS* top = sbt.getTop(); 
    ctx.setTop(top); 

    Frame frame(ctx.params); 

    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    ctx.uploadParams();  
    OPTIX_CHECK( optixLaunch( pip.pipeline, stream, ctx.d_param, sizeof( Params ), &(sbt.sbt), ctx.params->width, ctx.params->height, ctx.params->depth ) );
    CUDA_SYNC_CHECK();

    frame.download(); 

    bool yflip = false ; 
    frame.writePPM(outdir, "pixels.ppm", yflip );  
    int quality = Util::GetEValue<int>("QUALITY", 50); 
    frame.writeJPG(outdir, "pixels.jpg", quality);  
    frame.writeNP(  outdir, "posi.npy" );

    return 0 ; 
}
