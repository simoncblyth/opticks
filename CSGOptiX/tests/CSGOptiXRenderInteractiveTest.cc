/**
CSGOptiXRenderInteractiveTest.cc : Interactive raytrace rendering of analytic geometry 
========================================================================================

Analytic CSGOptiX rendering with SGLM/SGLFW interactive control/visualization. 
Usage with::

   ~/o/cx.sh 
   ~/o/CSGOptiX/cxr_min.sh 

Note this provides only CSGOptiX ray trace rendering, there is no rasterized toggle 
with "C" key to switch between CUDA and OpenGL rendering. 
For that functionality (but with triangulated geometry only) use:

   ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh 
   ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc 

TODO: 

* navigation frames currently managed in SScene(tri) but they 
  are equally relevant to ana+tri. 

  * Relocate the navigation frames, where ? Consider the workflow. 

* frame hopping like ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc 

* interactive vizmask control to hide/show geometry 

* WIP: enable mix and match ana/tri geometry, by incorporation
  of SScene and SOPTIX 

* imgui reincarnation 

* WIP: moved some common CSGOptiX stuff down to sysrap header-only such as jpg/png writing and image annotation 

* TODO: composite OpenGL event representation pixels together with OptiX ray traced renders

**/

#include "ssys.h"
#include "stree.h"
#include "schrono.h"
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "SGLFW.h"
#include "SGLFW_CUDA.h"

#include "SScene.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    static const char* _CXRI_REMOTE = "CSGOptiXRenderInteractiveTest__REMOTE" ;  
    bool CXRI_REMOTE = ssys::getenvbool(_CXRI_REMOTE); 
    bool is_remote_session = ssys::is_remote_session(); 

    if(is_remote_session && !CXRI_REMOTE )
    {
        std::cout << "main : ABORTING : as detected remote session from SSH_TTY SSH_CLIENT \n"; 
        std::cout << "to override : export " << _CXRI_REMOTE << "=1  ## warning have see gnome-shell crash with Wayland \n" ; 
        return 0 ;  
    }



    SEventConfig::SetRGModeRender(); 
    CSGFoundry* fd = CSGFoundry::Load(); 

    SScene* scene = fd->getScene(); 
    assert(!scene->is_empty()); 

    stree* st = fd->getTree(); 
    assert(st);
 
    const char* MOI = ssys::getenvvar("MOI", "0:0:-1" );  // default lvid 0 in remainder 
    const char* PFX = "EXTENT:" ;
    float extent = sstr::StartsWith(MOI, PFX) ? sstr::To<float>( MOI + strlen(PFX) ) : 0.f ;  
    // this extent handling is primarily for use with simple single solid test geometries

    sfr mfr = extent > 0.f ? sfr::MakeFromExtent<float>(extent) :  st->get_frame(MOI);    // HMM: what about when start from CSGMaker geometry ? 
    mfr.set_idx(-2);                 // maybe should start from stree/snode/sn geometry with an streemaker.h ?  
    // moved stree::get_frame prior to popping up the window, so failures 
    // from bad MOI dont cause hang 

    static const char* _FRAME_HOP = "CSGOptiXRenderInteractiveTest__FRAME_HOP" ;  
    static const char* _SGLM_DESC = "CSGOptiXRenderInteractiveTest__SGLM_DESC" ;  

    bool FRAME_HOP = ssys::getenvbool(_FRAME_HOP); 
    bool SGLM_DESC = ssys::getenvbool(_SGLM_DESC); 


    CSGOptiX* cx = CSGOptiX::Create(fd) ;

    SGLM& gm = *(cx->sglm) ; 
    SGLFW gl(gm); 
    SGLFW_CUDA interop(gm); 

    int sleep_break = ssys::getenvint("SLEEP_BREAK",0); 

    if(gl.level > 0) std::cout << "main:before loop  gl.get_wanted_frame_idx " <<  gl.get_wanted_frame_idx() << "\n" ; 

 
    while(gl.renderloop_proceed())
    {   
        gl.renderloop_head();


        // where to encapsulate this ? needs gl,gm,scene,mfr ?
        if(FRAME_HOP)
        {
            int wanted_frame_idx = gl.get_wanted_frame_idx() ; // -2 until press number key 0-9, back to -2 when press M  
            if(!gm.has_frame_idx(wanted_frame_idx) )
            {
                if(gl.level > 0) std::cout << "main:" << _FRAME_HOP << " wanted_frame_idx: " << wanted_frame_idx << "\n"; 
                if( wanted_frame_idx == -2 )
                { 
                    gm.set_frame(mfr);  
                    if(SGLM_DESC) std::cout 
                         << _SGLM_DESC << "\n"  
                         << gm.desc() 
                         ; 
                }
                else if( wanted_frame_idx >= 0 )
                { 
                    assert(scene); 
                    sfr wfr = scene->getFrame(wanted_frame_idx) ; 
                    gm.set_frame(wfr);   
                }
            }
        }


        uchar4* d_pixels = interop.output_buffer->map() ; 
        // d_pixels : device side pointer to output, 
        // mapping passes "baton" to CUDA/OptiX for filling    

        cx->setExternalDevicePixels(d_pixels); 
        cx->render_launch();     // ray tracing OptiX launch populating the pixels 


        // TODO: try moving wanted snap stuff to SGLM:gm not SGLFW:gl ?  
        //       gl provides the user choice, everything else to gm ?
        //       then could hide most of frame hopping detail 

        int wanted_snap = gl.get_wanted_snap();
        if( wanted_snap == 1 || wanted_snap == 2 )
        {
            std::cout << " gl.get_wanted_snap calling cx->render_snap \n" ; 
            switch(wanted_snap)
            {
                case 1: cx->render_save()          ; break ; 
                case 2: cx->render_save_inverted() ; break ; 
            }   
            gl.set_wanted_snap(0); 
        }

        interop.output_buffer->unmap() ;   // unmap, pass baton back to OpenGL for display 
        interop.displayOutputBuffer(gl.window);

        gl.renderloop_tail();
        if(sleep_break == 1) 
        { 
            schrono::sleep(1);  // 1 second then exit
            break ; 
        } 
    }   
    return 0 ; 
}

