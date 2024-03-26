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

int main()
{
    SScene* scene = SScene::Load("$SCENE_FOLD/scene"); 

    sframe fr ; 
    fr.ce = make_float4(0.f, 0.f, 0.f, 1000.f); 

    SGLM gm ; 
    gm.set_frame(fr) ; 
    // TODO: SGLM::set_ce/set_fr ? avoid heavyweight sframe when not needed ?

    return SGLFW_Scene::RenderLoop(scene, &gm) ; 
}

