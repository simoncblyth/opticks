/**
SGLFW_SOPTIX_Scene_test.cc 
============================

Started from SOPTIX_Scene_test.cc, a pure CUDA ppm render of optix triangles, 
added OpenGL interop viz for interactive view and parameter changing. 

Usage and impl::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh 
    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc

For a simpler non-encapsulated non-interactive OptiX only ppm render test, see:: 

    ~/o/sysrap/tests/SOPTIX_Scene_test.sh 
    ~/o/sysrap/tests/SOPTIX_Scene_test.cc

DONE: view maths for raytrace and rasterized now match each other quite closely 

**/

#include "ssys.h"
#include "SGLM.h"
#include "SScene.h"
#include "SOPTIX.h"
#include "SGLFW.h"
#include "SGLFW_Scene.h"

int main()
{
    bool DUMP = ssys::getenvbool("SGLFW_SOPTIX_Scene_test_DUMP"); 
    SScene* scene = SScene::Load("$SCENE_FOLD") ; 
    if(DUMP) std::cout << scene->desc() ; 
 
    SGLM gm ;
    SGLFW_Scene gls(scene, gm );
    SOPTIX      opx(scene, gm) ; 
    SGLFW_CUDA  interop(gm);    // interop buffer display coordination  

    SGLFW* gl = gls.gl ; 

    while(gl->renderloop_proceed())
    {   
        gl->renderloop_head();

        int wanted_frame_idx = gl->get_wanted_frame_idx() ;
        if(!gm.has_frame_idx(wanted_frame_idx) )
        {
            sfr wfr = scene->getFrame(wanted_frame_idx) ; 
            gm.set_frame(wfr);   
        } 

        if( gl->toggle.cuda )
        {
            uchar4* d_pixels = interop.output_buffer->map() ; // map buffer : for CUDA access 
            opx.render(d_pixels); 
            interop.output_buffer->unmap() ;                  // unmap buffer : access back to OpenGL
            interop.displayOutputBuffer(gl->window);
        }
        else
        { 
            gls.render(); 
        } 

        int wanted_snap = gl->get_wanted_snap();
        if( wanted_snap == 1 || wanted_snap == 2 )
        {
            std::cout << "wanted_snap " << wanted_snap << "\n" ; 
            bool yflip = wanted_snap == 1 ; 
            gl->snap_local(yflip);  
            gl->set_wanted_snap(0);
        }

        gl->renderloop_tail();
    }   
    return 0 ; 
}

