/**
CSGOptiXSimulateTest
======================

**/

#include <cuda_runtime.h>

#include "scuda.h"
#include "sqat4.h"
//#include "stran.h"
#include "NP.hh"

#include "SSys.hh"
#include "SPath.hh"
#include "SOpticks.hh"
#include "OPTICKS_LOG.hh"
#include "SOpticksResource.hh"
#include "SEvent.hh"
#include "SEventConfig.hh"
#include "SRG.h"
#include "OpticksGenstep.h"

#include "CSGFoundry.h"
#include "CSGGenstep.h"
#include "CSGOptiX.h"

#include "QSim.hh"
#include "QEvent.hh"

#include "Opticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv ); 
    ok.configure(); 

    const char* NAME = "CSGOptiXSimulateTest" ; 
    const char* cfbase = SOpticksResource::CFBase(); 
    const char* outfold = SEventConfig::OutFold();  
    LOG(info) << " outfold [" << outfold << "]"  ; 

    SOpticks::WriteOutputDirScript(outfold) ; // writes CSGOptiXSimulateTest_OUTPUT_DIR.sh in PWD 

    const char* idpath = SOpticksResource::IDPath();  // HMM: using OPTICKS_KEY geocache, not CSGFoundry
    const char* rindexpath = SPath::Resolve(idpath, "GScintillatorLib/LS_ori/RINDEX.npy", 0 );  

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    if(fd->hasMeta()) LOG(info) << "fd.meta\n" << fd->meta ; 
    //fd->upload(); 
    LOG(info) << fd->descComp(); 

    // GPU physics uploads : boundary+scintillation textures, property+randomState arrays    
    QSim<float>::UploadComponents(fd->icdf, fd->bnd, fd->optical, rindexpath ); 

    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    const char* cfbase_local = SSys::getenvvar("CFBASE_LOCAL") ; 
    assert(cfbase_local) ; 

    LOG(fatal) << "MIXING CSGFoundry combining basis cfbase with cfbase_local "; 
    std::cout << std::setw(20) << "cfbase" << ":" << cfbase << std::endl ; 
    std::cout << std::setw(20) << "cfbase_local" << ":" << cfbase_local  << std::endl ; 

    CSGFoundry* fdl = CSGFoundry::Load(cfbase_local, "CSGFoundry") ; 
    fdl->upload(); 

    CSGOptiX cx(&ok, fdl ); 
    float4 ce = make_float4( 0.f, 0.f, 0.f, 100.f );  
    cx.setComposition(ce); 
      
    quad6 gs ; 
    gs.q0.u = make_uint4( OpticksGenstep_PHOTON_CARRIER, 0u, 0u, 10u );   
    gs.q1.u = make_uint4( 0u,0u,0u,0u ); 
    gs.q2.f = make_float4( 0.f, 0.f, 0.f, 0.f );   // post
    gs.q3.f = make_float4( 1.f, 0.f, 0.f, 1.f );   // dirw
    gs.q4.f = make_float4( 0.f, 1.f, 0.f, 500.f ); // polw
    gs.q5.f = make_float4( 0.f, 0.f, 0.f, 0.f );   // flag 

    cx.setGenstep(&gs, 1); 
    cx.simulate();  

    const char* odir = SPath::Resolve(cfbase_local, "CSGOptiXSimulateTest", DIRPATH ); 
    LOG(info) << " odir " << odir ; 
    cx.event->save(odir); 
 
    cudaDeviceSynchronize(); 
    return 0 ; 
}
