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


struct SGLFW_SOPTIX_Scene_test
{
    bool DUMP ;
    stree* tree ;
    SScene* scene ;
    bool loaded ;
    SRecord* ar ;
    SRecord* br ;
    SGen* gs ;

   static bool LoadedGeom(const stree* t, const SScene* s)
   {
        if( t == nullptr || s == nullptr ) std::cerr
             << "SGLFW_SOPTIX_Scene_test::LoadedGEOM"
             << " LOAD FAIL "
             << " tree " << ( t ? "YES" : "NO " )
             << " scene " << ( s ? "YES" : "NO " )
             << "\n"
             ;

        //if (!t || !s) throw std::runtime_error("Failed to load tree or scene");
        if (!t || !s) return false ;
        return true;
    }

    SGLFW_SOPTIX_Scene_test();
};


/**
HMM: ELV selection can now reduce what is loaded into SScene
BUT there is no such functionality for stree ?
Is there potential for the inconsistency to cause issues ?
**/

SGLFW_SOPTIX_Scene_test::SGLFW_SOPTIX_Scene_test()
    :
    DUMP(ssys::getenvbool("SGLFW_SOPTIX_Scene_test_DUMP")),
    tree(stree::Load()),
    scene(SScene::Load()),
    loaded(LoadedGeom(tree,scene)),
    ar(loaded ? SRecord::Load("$AFOLD", "$AFOLD_RECORD_SLICE", "AFOLD_RECORD_TNUDGE") : nullptr),
    br(loaded ? SRecord::Load("$BFOLD", "$BFOLD_RECORD_SLICE", "BFOLD_RECORD_TNUDGE") : nullptr),
    gs(loaded ? SGen::Load("$AFOLD", "$AFOLD_GENSTEP_SLICE" ) : nullptr)
{
    if(DUMP && loaded) std::cout << scene->desc() ;
}


int main(int argc, char** argv)
{

    SGLFW_SOPTIX_Scene_test t ;
    if(!t.loaded) return 0;

    SGLM gm ;
    gm.setTreeScene(t.tree, t.scene);
    gm.setRecord( t.ar, t.br );
    gm.setGenstep( t.gs );


    if(ssys::is_under_ctest())
    {
        std::cout << argv[0] << " detected ssys::is_under_ctest so skip interactive renderloop popping up a window\n" ;
        return 0;
    }


    SGLFW gl(gm);

    SGLFW_Scene  sc(gl);
    SGLFW_Evt    ev(gl);

    std::cout << "[SGLFW_SOPTIX \n" ;
    SGLFW_SOPTIX ox(gl);
    std::cout << "]SGLFW_SOPTIX \n" ;

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

