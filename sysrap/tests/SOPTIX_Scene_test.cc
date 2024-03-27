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

#include "ssys.h"
#include "spath.h"
#include "scuda.h"
#include "sppm.h"

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
 
    int ihandle = ssys::getenvint("HANDLE", 0)  ; 
    float extent = 10000.f ;
    switch(ihandle)
    {   
        case -1: extent = 12000.f ; break ; 
        case  0: extent = 12000.f ; break ; 
        case  1: extent = 100.f ; break ; 
        case  2: extent = 500.f ; break ; 
        case  3: extent = 500.f ; break ; 
        case  4: extent = 500.f ; break ; 
        case  5: extent = 100.f ; break ; 
        case  6: extent = 200.f ; break ; 
        case  7: extent = 500.f ; break ; 
        case  8: extent = 500.f ; break ; 
    } 

 
    sfr fr ; 
    fr.set_extent( extent ); 
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

    
    uchar4* d_pixels = nullptr ;
    size_t num_pixel = gm.Width()*gm.Height(); 
    size_t pixel_bytes = num_pixel*sizeof(uchar4) ; 
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_pixels ), pixel_bytes )); 
    uchar4* pixels = new uchar4[num_pixel] ; 

    SOPTIX_Params par ; ; 

    par.width = gm.Width() ; 
    par.height = gm.Height() ; 
    par.pixels = d_pixels ; 
    par.tmin = 0.1f ; 
    par.tmax = 1e9f ; 
    par.cameratype = gm.cam ; 
    SGLM::Copy(&par.eye.x, gm.e ); 
    SGLM::Copy(&par.U.x  , gm.u );  
    SGLM::Copy(&par.V.x  , gm.v );  
    SGLM::Copy(&par.W.x  , gm.w );  

    par.handle = ihandle == -1 ? scn.ias->handle : scn.meshgroup[ihandle]->gas->handle ;  

    SOPTIX_Params* d_param = par.device_alloc(); 
    par.upload(d_param); 


    OPTIX_CHECK( optixLaunch(
                 pip.pipeline,
                 0,             // stream
                 (CUdeviceptr)d_param,
                 sizeof( SOPTIX_Params ),
                 &(sbt.sbt),
                 gm.Width(),  // launch width
                 gm.Height(), // launch height
                 1            // launch depth
                 ) );
    
    CUDA_SYNC_CHECK();
    CUDA_CHECK( cudaMemcpy( pixels, reinterpret_cast<void*>(d_pixels), pixel_bytes, cudaMemcpyDeviceToHost ));
     
    const char* ppm_path = getenv("PPM_PATH") ;   
    bool yflip = false ; 
    sppm::Write(ppm_path, gm.Width(), gm.Height(), 4, (unsigned char*)pixels, yflip );  


    return 0 ; 
}
