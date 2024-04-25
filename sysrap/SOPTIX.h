#pragma once
/**
SOPTIX.h : top level coordinator of triangulated raytrace render
=================================================================

**/

#include "ssys.h"
#include "SOPTIX_Context.h"
#include "SOPTIX_Desc.h"
#include "SOPTIX_MeshGroup.h"
#include "SOPTIX_Scene.h"
#include "SOPTIX_Module.h"
#include "SOPTIX_Pipeline.h"
#include "SOPTIX_SBT.h"
#include "SOPTIX_Params.h"
#include "SOPTIX_Pixels.h"


struct SOPTIX
{  
    SGLM&           gm ; 

    SOPTIX_Context  ctx ; 
    SOPTIX_Options  opt ;  
    SOPTIX_Module   mod ; 
    SOPTIX_Pipeline pip ; 
    SOPTIX_Scene    scn ;  
    SOPTIX_SBT      sbt ;
    SOPTIX_Params   par = {} ;
    SOPTIX_Params*  d_param ; 

    CUstream stream = 0 ; 
    unsigned depth = 1 ; 
 
    int HANDLE ; 
    OptixTraversableHandle handle ;

    SOPTIX_Pixels*  pix ;   // optional internally managed pixels

    SOPTIX(const SScene* _scn, SGLM& _gm );  
    void set_param(uchar4* d_pixels);
    void render(uchar4* d_pixels);
    void render_ppm(const char* _path);

};


inline SOPTIX::SOPTIX(const SScene* _scn, SGLM& _gm)
    :
    gm(_gm),
    mod(ctx.context, opt, "$SOPTIX_PTX" ),
    pip(ctx.context, mod.module, opt ),
    scn(&ctx, _scn ),
    sbt(pip, scn ),
    d_param(SOPTIX_Params::DeviceAlloc()),
    HANDLE(ssys::getenvint("HANDLE", -1)),
    handle(scn.getHandle(HANDLE)),
    pix(nullptr)
{
}

inline void SOPTIX::set_param(uchar4* d_pixels)
{
    par.width = gm.Width() ; 
    par.height = gm.Height() ; 
    par.pixels = d_pixels ; 
    par.tmin = gm.get_near_abs() ; 
    par.tmax = gm.get_far_abs() ; 
    par.cameratype = gm.cam ; 
    par.visibilityMask = gm.vizmask ; 

    SGLM::Copy(&par.eye.x, gm.e ); 
    SGLM::Copy(&par.U.x  , gm.u );  
    SGLM::Copy(&par.V.x  , gm.v );  
    SGLM::Copy(&par.W.x  , gm.w );  
    par.handle = handle ; 

    par.upload(d_param); 
}

inline void SOPTIX::render(uchar4* d_pixels)
{
    set_param(d_pixels); 

    OPTIX_CHECK( optixLaunch(
                 pip.pipeline,
                 stream,
                 (CUdeviceptr)d_param,
                 sizeof( SOPTIX_Params ),
                 &(sbt.sbt),
                 gm.Width(),  
                 gm.Height(), 
                 depth
                 ) );
 
    CUDA_SYNC_CHECK();
}


inline void SOPTIX::render_ppm(const char* _path)
{
    const char* path = spath::Resolve(_path); 

    std::cout << "SOPTIX::render_ppm [" << ( path ? path : "-" ) << "]\n" ; 

    if(!pix) pix = new SOPTIX_Pixels(gm) ;
    render(pix->d_pixels); 

    pix->download(); 

    pix->save_ppm(path); 



}
