/**
SGLFW_SOPTIX_Scene_test.cc
============================

Started from SOPTIX_Scene_test.cc, a pure CUDA ppm render of optix triangles,
added OpenGL interop viz for interactive view and parameter changing.

Usage and impl::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc

As this and alias are now installed can just use::

    SGLFW_SOPTIX_Scene_test.sh
    ssst.sh

For a simpler non-encapsulated non-interactive OptiX only ppm render test, see::

    ~/o/sysrap/tests/SOPTIX_Scene_test.sh
    ~/o/sysrap/tests/SOPTIX_Scene_test.cc

DONE: view maths for raytrace and rasterized now match each other quite closely

**/

#include "ssys.h"
#include "SGLM.h"
#include "SScene.h"

#include "SGLFW.h"
#include "SGLFW_Scene.h"
#include "SGLFW_Evt.h"
#include "SGLFW_SOPTIX.h"


int main(int argc, char** argv)
{
    bool DUMP = ssys::getenvbool("SGLFW_SOPTIX_Scene_test_DUMP");

    stree* tree = stree::Load();
    SScene* scene = SScene::Load() ;
    // HMM: ELV selection can now reduce what is loaded into SScene
    // BUT there is no such functionality for stree ?
    // There is potential for the inconsistency to cause issues ?
    if( tree == nullptr || scene == nullptr ) std::cerr
         << "SGLFW_SOPTIX_Scene_test.main"
         << " LOAD FAIL "
         << " tree " << ( tree ? "YES" : "NO " )
         << " scene " << ( scene ? "YES" : "NO " )
         << "\n"
         ;

    if( tree == nullptr || scene == nullptr ) return 0;

    if(DUMP) std::cout << scene->desc() ;

    SRecord* ar = SRecord::Load("$AFOLD", "$AFOLD_RECORD_SLICE" ) ;
    SRecord* br = SRecord::Load("$BFOLD", "$BFOLD_RECORD_SLICE" ) ;

    SGLM gm ;
    gm.setTreeScene(tree, scene);
    gm.setRecord( ar, br );

    SGLFW gl(gm);

    SGLFW_Scene  sc(gl);
    SGLFW_Evt    ev(gl);
    SGLFW_SOPTIX ox(gl);

    if(ssys::is_under_ctest())
    {
        std::cout << argv[0] << " detected ssys::is_under_ctest so skip interactive renderloop popping up a window\n" ;
        return 0;
    }

    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();
        gl.handle_frame_hop();

        if(gm.option.M)
        {
            if( gm.toggle.cuda ) ox.render();
            else                 sc.render();
        }
        ev.render();

        gl.handle_snap();
        gl.renderloop_tail();
    }
    return 0 ;
}

