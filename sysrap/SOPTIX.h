#pragma once
/**
SOPTIX.h : top level coordinator of triangulated raytrace render
=================================================================

envvars
--------

SOPTIX__HANDLE



**/

#include "ssys.h"
#include "SOPTIX_Context.h"
#include "SOPTIX_Desc.h"
#include "SOPTIX_MeshGroup.h"
#include "SOPTIX_Accel.h"
#include "SOPTIX_Scene.h"
#include "SOPTIX_Module.h"
#include "SOPTIX_Pipeline.h"
#include "SOPTIX_SBT.h"
#include "SOPTIX_Params.h"
#include "SOPTIX_Pixels.h"
#include "SOPTIX_Options.h"


struct SOPTIX
{
    static constexpr const char* __HANDLE = "SOPTIX__HANDLE" ;
    static int  Initialize();

    int             irc ;
    SGLM&           gm  ;
    const char*     _optixpath ;
    const char*     optixpath ;

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

    SOPTIX(SGLM& _gm );
    void init();

    void set_param(uchar4* d_pixels);
    void render(uchar4* d_pixels);
    void render_ppm(const char* _path);

    std::string desc() const ;
};


inline int SOPTIX::Initialize()
{
    if(SOPTIX_Options::Level() > 0) std::cout << "[SOPTIX::Initialize\n" ;
    return 0 ;
}


inline SOPTIX::SOPTIX(SGLM& _gm)
    :
    irc(Initialize()),
    gm(_gm),
#ifdef CONFIG_Debug
    _optixpath("${SOPTIX__optixpath:-$OPTICKS_PREFIX/optix/objects-Debug/SysRap_OPTIX/SOPTIX.ptx}"),
#elif CONFIG_Release
    _optixpath("${SOPTIX__optixpath:-$OPTICKS_PREFIX/optix/objects-Release/SysRap_OPTIX/SOPTIX.ptx}"),
#else
    _optixpath(nullptr),
#endif
    optixpath(_optixpath ? spath::Resolve(_optixpath) : nullptr),
    mod(ctx.context, opt, optixpath ),
    pip(ctx.context, mod.module, opt ),
    scn(&ctx, gm.scene ),
    sbt(pip, scn ),
    d_param(SOPTIX_Params::DeviceAlloc()),
    HANDLE(ssys::getenvint(__HANDLE, -1)),
    handle(scn.getHandle(HANDLE)),
    pix(nullptr)
{
    init();
}

inline void SOPTIX::init()
{
    if(SOPTIX_Options::Level() > 0)  std::cout << "]SOPTIX::init \n" ;
    if(SOPTIX_Options::Level() > 1)  std::cout << desc() ;
}


/**
SOPTIX::set_param
------------------

cf CSGOptiX::prepareParamRender

**/


inline void SOPTIX::set_param(uchar4* d_pixels)
{
    par.width = gm.Width() ;
    par.height = gm.Height() ;
    par.pixels = d_pixels ;
    par.tmin = gm.get_near_abs() ;
    par.tmax = gm.get_far_abs() ;
    par.cameratype = gm.cam ;
    par.vizmask = gm.vizmask ;

    SGLM::Copy(&par.eye.x, gm.e );
    SGLM::Copy(&par.U.x  , gm.u );
    SGLM::Copy(&par.V.x  , gm.v );
    SGLM::Copy(&par.W.x  , gm.w );

    SGLM::Copy(&par.WNORM.x, gm.wnorm );
    SGLM::Copy(&par.ZPROJ.x, gm.zproj );


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


inline std::string SOPTIX::desc() const
{
    std::stringstream ss ;
    ss
        << "[SOPTIX::desc\n"
        << " _optixpath [" << ( _optixpath ? _optixpath : "-" ) << "]\n"
        << " optixpath  [" << ( optixpath ? optixpath : "-" ) << "]\n"
        << " [" << __HANDLE << "] " << HANDLE << "\n"
        << "]SOPTIX::desc\n"
        ;
    std::string str = ss.str() ;
    return str ;
}

