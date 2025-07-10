/**
examples/UseShaderSGLFW_SScene_encapsulated.cc
===============================================

::

    ~/o/examples/UseShaderSGLFW_SScene_encapsulated/go.sh

See also::

    TEST=CreateFromTree ~/o/sysrap/tests/SScene_test.sh
    TEST=Load           ~/o/sysrap/tests/SScene_test.sh

    ~/o/u4/tests/U4TreeCreateTest.sh

    ~/o/u4/tests/U4Mesh_test.sh
    ~/o/sysrap/tests/SMesh_test.sh

**/

#include "SGLFW_Scene.h"

sn::POOL* sn::pool = nullptr ;
s_pa::POOL* s_pa::pool = nullptr ;
s_tv::POOL* s_tv::pool = nullptr ;
s_bb::POOL* s_bb::pool = nullptr ;
s_csg* s_csg::INSTANCE = nullptr ;


int main()
{
    stree* tree = stree::Load();
    SScene* scene = SScene::Load_();

    sfr fr ;
    fr.set_extent(10000.);

    SGLM gm ;
    gm.setTreeScene(tree, scene);
    gm.set_frame(fr) ;

    SGLFW gl(gm);
    SGLFW_Scene  sc(gl);
    sc.renderloop();

    return 0;
}

