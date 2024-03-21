/**
SOPTIX_Scene_test.sh 
=======================

::
 
    ~/o/sysrap/tests/SOPTIX_Scene_test.sh 
    ~/o/sysrap/tests/SOPTIX_Scene_test.cc

Related::

    ~/o/sysrap/tests/SCUDA_Mesh_test.cc
    ~/o/sysrap/SOPTIX_Mesh.h

**/

#include "scuda.h"
#include "SScene.h"

#include "SCUDA_Mesh.h"

#include "SOPTIX.h"
#include "SOPTIX_Mesh.h"
#include "SOPTIX_Scene.h"

int main()
{
    SScene* _sc = SScene::Load("$SCENE_FOLD/scene") ; 
    std::cout << _sc->desc() ; 
  
    SOPTIX ox ; 
    std::cout << ox.desc() ; 

    SOPTIX_Scene sc(&ox, _sc );  
    std::cout << sc.desc() ; 

    return 0 ; 
}
