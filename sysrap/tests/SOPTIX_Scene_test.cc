/**
SOPTIX_Scene_test.cc 
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

#include "SGLM.h"
#include "SScene.h"

#include "SOPTIX.h"

#include "SCUDA_MeshGroup.h"
#include "SOPTIX_MeshGroup.h"
#include "SOPTIX_Scene.h"
#include "SOPTIX_Module.h"
#include "SOPTIX_Pipeline.h"
#include "SOPTIX_SBT.h"

#include "SOPTIX_Params.h"


int main()
{
    bool dump = false ; 

    SScene* _scn = SScene::Load("$SCENE_FOLD/scene") ; 
    if(dump) std::cout << _scn->desc() ; 
  
    sframe fr ; 
    fr.ce = make_float4(0.f, 0.f, 0.f, 1000.f); 
    // TODO: determine CE from scene and view options 

    SGLM gm ; 
    gm.set_frame(fr) ; 
    //std::cout << gm.desc() ;  


    SOPTIX opx ; 
    if(dump) std::cout << opx.desc() ; 

    SOPTIX_Options opt ;  
    if(dump) std::cout << opt.desc() ; 

    SOPTIX_Scene scn(&opx, _scn );  
    if(dump) std::cout << scn.desc() ; 

    SOPTIX_Module mod(opx.context, opt, "$SOPTIX_PTX" ); 
    if(dump) std::cout << mod.desc() ; 

    SOPTIX_Pipeline pip(opx.context, mod.module, opt ); 
    if(dump) std::cout << pip.desc() ; 

    SOPTIX_SBT sbt(pip, scn );
    if(dump) std::cout << sbt.desc() ; 


    // before complicating things with OpenGL interop, test pure CUDA
    //SCUDAOutputBuffer<uchar4>( SCUDAOutputBufferType::GL_INTEROP, gm.Width(), gm.Height() );  
    // need img handling like qudarap/tests/QTexRotateTest.cc with SIMG 
    //  examples/UseOptiX7GeometryInstanced/Engine.cc


    SOPTIX_Params par ; ; 
    par.device_alloc(); 
    par.width = gm.Width() ; 
    par.height = gm.Height() ; 
    par.pixels = nullptr ; 
    par.tmin = 0.f ; 
    par.tmax = 1e99f ; 
    par.cameratype = 0 ; 
    SGLM::Copy(&par.eye.x, gm.e ); 
    SGLM::Copy(&par.U.x  , gm.u );  
    SGLM::Copy(&par.V.x  , gm.v );  
    SGLM::Copy(&par.W.x  , gm.w );  
    par.handle = scn.ias->handle ;  
    par.upload(); 



    return 0 ; 
}
