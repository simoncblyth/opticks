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
#include "SGLFW_Evt.h"
#include "SGLFW_Scene.h"


int main()
{
    bool DUMP = ssys::getenvbool("SGLFW_SOPTIX_Scene_test_DUMP");

    const char* ss = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim") ;
    SScene* scene = SScene::Load(ss) ;
    stree* tree = stree::Load(ss);
    if(DUMP) std::cout << scene->desc() ;

    SRecordInfo* ar = SRecordInfo::Load("$AFOLD/record.npy") ;
    SRecordInfo* br = SRecordInfo::Load("$BFOLD/record.npy") ;


    SGLM gm ;
    gm.setTreeScene(tree, scene);
    gm.setRecordInfo( ar, br ); 

    SGLFW gl(gm); 

    SGLFW_Scene glsc(gl);
    SGLFW_Evt   glev(gl); 

    SOPTIX      opx(gm) ;
    SGLFW_CUDA  interop(gm);    // interop buffer display coordination

    bool first = true ;  ; 

    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();
        gl.handle_frame_hop();

        if( gm.toggle.cuda )
        {
            uchar4* d_pixels = interop.output_buffer->map() ; // map buffer : give access to CUDA
            opx.render(d_pixels);
            if(first)
            {
                first = false ;  
                const char* path = "/tmp/out.ppm";  
                std::cout << " opx.render_ppm path [" << path << "]\n" ;   
                opx.render_ppm(path); 
            }
            interop.output_buffer->unmap() ;                  // unmap buffer : end CUDA access, back to OpenGL
            interop.displayOutputBuffer(gl.window);     
        }
        else
        {
            glsc.render();
        }
        glev.render(); 

        gl.handle_snap(); 
        gl.renderloop_tail();
    }
    return 0 ;
}

