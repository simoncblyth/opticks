/**
SGLFW_SOPTIX_Scene_test.cc 
============================

Started from SOPTIX_Scene_test.cc, a pure CUDA ppm render of optix triangles, 
added OpenGL interop viz for interactive view and parameter changing. 

Usage and impl::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh 
    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc

Related::

    ~/o/sysrap/tests/SOPTIX_Scene_test.sh 
    ~/o/sysrap/tests/SOPTIX_Scene_test.cc


TODO: get view maths for raytrace and rasterized to match each other 

* HMM target control would help with that 

**/

#include "ssys.h"
#include "spath.h"
#include "scuda.h"

#include "SGLM.h"
#include "SScene.h"

#include "SOPTIX.h"

#include "SCUDA_MeshGroup.h"
#include "SOPTIX_MeshGroup.h"
#include "SOPTIX_Scene.h"
#include "SOPTIX_Module.h"
#include "SOPTIX_Pipeline.h"
#include "SOPTIX_SBT.h"

#include "SOPTIX_Params.h"

#include "SGLFW.h"
#include "SGLFW_Scene.h"


float kludge_extent(int ihandle)
{
    float extent = 10000.f ;
    switch(ihandle)
    {   
        case -1: extent = 12000.f ; break ; 
        case  0: extent = 12000.f ; break ; 
        case  1: extent = 100.f ; break ; 
        case  2: extent = 500.f ; break ; 
        case  3: extent = 500.f ; break ; 
        case  4: extent = 500.f ; break ; 
        case  5: extent = 100.f ; break ; 
        case  6: extent = 200.f ; break ; 
        case  7: extent = 500.f ; break ; 
        case  8: extent = 500.f ; break ; 
    } 
    return extent ; 
}




int main()
{
    bool DUMP = ssys::getenvbool("SGLFW_SOPTIX_Scene_test_DUMP"); 
    bool dump = false ; 

    SScene* _scn = SScene::Load("$SCENE_FOLD") ; 
    if(DUMP) std::cout << _scn->desc() ; 
 
    int ihandle = ssys::getenvint("HANDLE", 0)  ; 
    int FRAME = ssys::getenvint("FRAME", 0)  ; 
    int num_frame = _scn->frame.size(); 
    const float* _ce = _scn->get_ce(0) ; 

    std::cout 
         << "num_frame " << num_frame 
         << "FRAME " << FRAME
         << " _ce[3] " << ( _ce ? _ce[3] : -1.f )    
         << "\n" ; 
  

    sfr fr = ( FRAME > -1 && FRAME < num_frame ) ? _scn->frame[FRAME] : sfr::MakeFromCE(_ce) ; 
    // TODO: interactive frame jumping ?


    SGLM gm ; 
    gm.set_frame(fr) ; 

    SOPTIX opx ; 
    if(dump) std::cout << opx.desc() ; 

    SOPTIX_Options opt ;  
    if(dump) std::cout << opt.desc() ; 

    SOPTIX_Scene scn(&opx, _scn );  
    if(dump) std::cout << scn.desc() ; 

    SOPTIX_Module mod(opx.context, opt, "$SOPTIX_PTX" ); 
    if(dump) std::cout << mod.desc() ; 

    SOPTIX_Pipeline pip(opx.context, mod.module, opt ); 
    if(dump) std::cout << pip.desc() ; 

    SOPTIX_SBT sbt(pip, scn );
    if(dump) std::cout << sbt.desc() ; 
  
 
    SGLFW_Scene glsc(_scn, &gm );
    SGLFW* gl = glsc.gl ; 

    SGLFW_CUDA interop(gm); 
 
    SOPTIX_Params par ; ; 
    SOPTIX_Params* d_param = par.device_alloc(); 

    CUstream stream = 0 ; 
    unsigned depth = 1 ; 

    while(gl->renderloop_proceed())
    {   
        gl->renderloop_head();

        if( gl->toggle.cuda )
        {
            uchar4* d_pixels = interop.output_buffer->map() ; 
       
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
            par.handle = ihandle == -1 ? scn.ias->handle : scn.meshgroup[ihandle]->gas->handle ;  
            par.upload(d_param); 

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
            interop.output_buffer->unmap() ; 

            interop.displayOutputBuffer(gl->window);
        }
        else
        { 
            glsc.render(); 
        }
        gl->renderloop_tail();
    }   
    return 0 ; 
}
