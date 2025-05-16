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

    const char* ss = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim") ;
    SScene* scene = SScene::Load(ss) ;
    stree* tree = stree::Load(ss);
    if(DUMP) std::cout << scene->desc() ;

    //std::cout << "[SGLM\n" ; 
    SGLM gm ;
    gm.setTreeScene(tree, scene);
    //std::cout << "]SGLM\n" ; 

    //std::cout << "[SGLFW_Scene\n" ; 
    SGLFW_Scene gls(scene, gm );
    //std::cout << "]SGLFW_Scene\n" ; 

    //std::cout << "[SOPTIX\n" ; 
    SOPTIX      opx(scene, gm) ;
    //std::cout << "]SOPTIX\n" ; 

    std::cout << "[SGLFW_CUDA\n" ; 
    SGLFW_CUDA  interop(gm);    // interop buffer display coordination
    std::cout << "]SGLFW_CUDA\n" ; 

    SGLFW* gl = gls.gl ;
    bool first = true ;  ; 

    while(gl->renderloop_proceed())
    {
        gl->renderloop_head();
        gl->handle_frame_hop();

        if( gl->toggle.cuda )
        {
            uchar4* d_pixels = interop.output_buffer->map() ; // map buffer : for CUDA access
            opx.render(d_pixels);

            if(first)
            {
                first = false ;  
                const char* path = "/tmp/out.ppm";  
                std::cout << " opx.render_ppm path [" << path << "]\n" ;   
                opx.render_ppm(path); 
            }

            interop.output_buffer->unmap() ;                  // unmap buffer : access back to OpenGL
            interop.displayOutputBuffer(gl->window);
        }
        else
        {
            gls.render();
        }

        gl->handle_snap(); 
        gl->renderloop_tail();
    }
    return 0 ;
}

