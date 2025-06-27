/**
SGLFW_Evt_test.cc
============================

Started from SOPTIX_Scene_test.cc, a pure CUDA ppm render of optix triangles,
added OpenGL interop viz for interactive view and parameter changing.

Usage and impl::

    ~/o/sysrap/tests/SGLFW_Evt_test.sh
    ~/o/sysrap/tests/SGLFW_Evt_test.cc

For a simpler non-encapsulated non-interactive OptiX only ppm render test, see::

    ~/o/sysrap/tests/SOPTIX_Scene_test.sh
    ~/o/sysrap/tests/SOPTIX_Scene_test.cc

DONE: view maths for raytrace and rasterized now match each other quite closely

**/

#include "ssys.h"
#include "spath.h"
#include "SGLM.h"
#include "stree.h"
#include "SScene.h"
#include "SRecord.h"
#include "SGLFW.h"
#include "SGLFW_Scene.h"
#include "SGLFW_Evt.h"

int main()
{
    bool DUMP = ssys::getenvbool("SGLFW_Evt_test_DUMP");

    const char* ss = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim") ;
    SScene* scene = SScene::Load(ss) ;
    stree* tree = stree::Load(ss);
    if(DUMP) std::cout << scene->desc() ;

    SRecord* ar = SRecord::Load("$AFOLD") ;
    SRecord* br = SRecord::Load("$BFOLD") ;

    if(DUMP) std::cout
         << " ar " << ( ar ? ar->desc() : "-" ) << "\n"
         << " br " << ( br ? br->desc() : "-" ) << "\n"
         ;


    SGLM gm ;
    gm.setTreeScene(tree, scene);
    gm.setRecord( ar, br );

    SGLFW gl(gm);
    SGLFW_Evt   ev(gl);
    SGLFW_Scene sc(gl);

    while(gl.renderloop_proceed())
    {
        gl.renderloop_head();
        gl.handle_frame_hop();

        if(gm.option.M) sc.render();
        ev.render();

        gl.renderloop_tail();
    }
    return 0 ;
}

