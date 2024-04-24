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


DONE: get view maths for raytrace and rasterized to match each other 

* HMM target control would help with that 


How to encapsulate further ?
----------------------------

There is communication between SOPTIX_Params SGLM and interop output buffer
so maybe move more into SOPTIX_Scene. 

But to encompass the whole task need a level above SGLFW and SOPTIX

* SRender.h too similar to old SRenderer, maybe SVIZ.h  


**/

#include "ssys.h"
#include "spath.h"
#include "scuda.h"

#include "SGLM.h"
#include "SScene.h"

#include "SOPTIX_Context.h"
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




int main()
{
    bool DUMP = ssys::getenvbool("SGLFW_SOPTIX_Scene_test_DUMP"); 
    bool dump = false ; 

    SScene* _scn = SScene::Load("$SCENE_FOLD") ; 
    if(DUMP) std::cout << _scn->desc() ; 
 


    SOPTIX_Context ctx ; 
    if(dump) std::cout << ctx.desc() ; 

    SOPTIX_Options opt ;  
    if(dump) std::cout << opt.desc() ; 

    SOPTIX_Module mod(ctx.context, opt, "$SOPTIX_PTX" ); 
    if(dump) std::cout << mod.desc() ; 

    SOPTIX_Pipeline pip(ctx.context, mod.module, opt ); 
    if(dump) std::cout << pip.desc() ; 



    SOPTIX_Scene scn(&ctx, _scn );  
    if(dump) std::cout << scn.desc() ; 

    SOPTIX_SBT sbt(pip, scn );
    if(dump) std::cout << sbt.desc() ; 
  

    int HANDLE = ssys::getenvint("HANDLE", -1)  ; 
    OptixTraversableHandle handle = scn.getHandle(HANDLE) ;

 
    SGLM gm ;
    SGLFW_Scene glsc(_scn, &gm );


    SGLFW* gl = glsc.gl ; 
    gl->setCursorPos(0.f,0.f); 

    SGLFW_CUDA interop(gm); 
 
    SOPTIX_Params* d_param = SOPTIX_Params::DeviceAlloc(); 
    SOPTIX_Params par = {} ; 

    CUstream stream = 0 ; 
    unsigned depth = 1 ; 


    while(gl->renderloop_proceed())
    {   
        gl->renderloop_head();


        int wanted_frame_idx = gl->get_wanted_frame_idx() ;
        if(!gm.has_frame_idx(wanted_frame_idx) )
        {
            sfr wfr = _scn->getFrame(wanted_frame_idx) ; 
            gm.set_frame(wfr);   
        } 


        if( gl->toggle.cuda )
        {
            uchar4* d_pixels = interop.output_buffer->map() ; 

            // --
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

            // -- 
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
