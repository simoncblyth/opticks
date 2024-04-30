/**
CSGOptiXRenderInteractiveTest.cc : Interactive raytrace rendering of analytic geometry 
========================================================================================

Analytic CSGOptiX rendering with SGLM/SGLFW interactive control/visualization. 

Usage with::

   ~/o/CSGOptiX/cxr_min.sh 

This follows the approach of::

   ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh 
   ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc 

TODO: 

* navigation frames currently managed in SScene(tri) but they 
  are equally relevant to ana+tri. 

  * Relocate the navigation frames, where ? Consider the workflow. 

* frame hopping like ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc 

* interactive vizmask control to hide/show geometry 

* enable mix and match ana/tri geometry

  * will need to incorporate SScene and SOPTIX into CSGOptiX

* imgui reincarnation 

* move common CSGOptiX stuff down to sysrap header-only such as jpg/png writing and image annotation 

**/

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

    SEventConfig::SetRGModeRender(); 
    CSGFoundry* fd = CSGFoundry::Load(); 

    if(fd->getScene()->is_empty())
    {
        LOG(fatal) << "CSGFoundry::Load GIVES EMPTY SCENE : TRANSITIONAL KLUDGE : TRY TO LOAD FROM SCENE_FOLD " ; 
        SScene* _scene = SScene::Load("$SCENE_FOLD");   
        fd->setOverrideScene(_scene); 
    }


    CSGOptiX* cx = CSGOptiX::Create(fd) ;

    SGLM& gm = *(cx->sglm) ; 
    SGLFW gl(gm); 
    SGLFW_CUDA interop(gm); 
 
    while(gl.renderloop_proceed())
    {   
        gl.renderloop_head();

        uchar4* d_pixels = interop.output_buffer->map() ; 

        cx->setExternalDevicePixels(d_pixels); 
        cx->render_launch(); 

        interop.output_buffer->unmap() ; 
        interop.displayOutputBuffer(gl.window);

        gl.renderloop_tail();
    }   
    return 0 ; 
}

