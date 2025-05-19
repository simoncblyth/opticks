/**
SGLFW_Event_test.cc
============================

Started from SOPTIX_Scene_test.cc, a pure CUDA ppm render of optix triangles,
added OpenGL interop viz for interactive view and parameter changing.

Usage and impl::

    ~/o/sysrap/tests/SGLFW_Event_test.sh
    ~/o/sysrap/tests/SGLFW_Event_test.cc

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
#include "SRecordInfo.h"
#include "SGLFW.h"
#include "SGLFW_Event.h"

int main()
{
    bool DUMP = ssys::getenvbool("SGLFW_Event_test_DUMP");

    const char* ss = spath::Resolve("$CFBaseFromGEOM/CSGFoundry/SSim") ;
    SScene* scene = SScene::Load(ss) ;
    stree* tree = stree::Load(ss);
    if(DUMP) std::cout << scene->desc() ;

    SRecordInfo* ar = SRecordInfo::Load("$AFOLD/record.npy") ;
    SRecordInfo* br = SRecordInfo::Load("$BFOLD/record.npy") ;

    if(DUMP) std::cout
         << " ar " << ( ar ? ar->desc() : "-" ) << "\n"
         << " br " << ( br ? br->desc() : "-" ) << "\n"
         ;


    SGLM gm ;
    gm.setTreeScene(tree, scene);
    gm.setRecordInfo( ar, br ); 


    SGLFW_Event glsc(scene, gm );
    SGLFW* gl = glsc.gl ;

    while(gl->renderloop_proceed())
    {
        gl->renderloop_head();
        gl->handle_frame_hop();

        glsc.render();

        gl->renderloop_tail();
    }
    return 0 ;
}

