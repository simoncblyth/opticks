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
#include "SGLM.h"
#include "SScene.h"
#include "SRecordInfo.h"
#include "SGLFW.h"
#include "SGLFW_Event.h"

int main()
{
    bool DUMP = ssys::getenvbool("SGLFW_Event_test_DUMP"); 
    SScene* scene = SScene::Load("$SCENE_FOLD") ; 
    if(DUMP) std::cout << scene->desc() ; 
 
    SRecordInfo* sr = SRecordInfo::Load("$SRECORD_PATH") ; 

    SGLM gm ;
    SGLFW_Event glsc(scene, gm , sr);
    //SGLFW_Event glsc(nullptr, gm , sr);
    SGLFW* gl = glsc.gl ; 

    while(gl->renderloop_proceed())
    {   
        gl->renderloop_head();
        glsc.render(); 
        gl->renderloop_tail();
    }   
    return 0 ; 
}

