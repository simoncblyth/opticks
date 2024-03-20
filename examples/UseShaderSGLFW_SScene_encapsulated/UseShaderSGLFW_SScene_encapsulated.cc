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
    return SGLFW_Scene::RenderLoop(scene) ; 
}

