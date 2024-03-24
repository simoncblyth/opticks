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

#include "spath.h"
#include "scuda.h"
#include "SScene.h"

#include "SCUDA_Mesh.h"

#include "SOPTIX.h"
#include "SOPTIX_MeshGroup.h"
#include "SOPTIX_Scene.h"
#include "SOPTIX_Module.h"

int main()
{
    SScene* _scn = SScene::Load("$SCENE_FOLD/scene") ; 
    std::cout << _scn->desc() ; 
  
    SOPTIX opx ; 
    std::cout << opx.desc() ; 

    SOPTIX_Options opt ;  

    SOPTIX_Scene scn(&opx, _scn );  
    std::cout << scn.desc() ; 

    const char* _ptxpath = "$OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx" ; 
    // SHOULD BE: const char* _ptxpath = "$OPTICKS_PREFIX/ptx/sysrap_generated_SOPTIX.cu.ptx" ; 

    SOPTIX_Module mod(opx.context, opt,  _ptxpath ); 
    std::cout << mod.desc() ; 

    SOPTIX_Pipeline pip(opx.context, mod.module, opt ); 
    std::cout << pip.desc() ; 

    SOPTIX_SBT sbt( SOPTIX_Pipeline& pip, SOPTIX_Scene& scn );
    std::cout << sbt.desc() ; 


    return 0 ; 
}
