/**
CSGOptiXSimulateTest
======================

**/

#include <cuda_runtime.h>
#include <algorithm>
#include <csignal>
#include <iterator>

#include "scuda.h"
#include "sqat4.h"
#include "stran.h"
#include "NP.hh"

#include "SSys.hh"
#include "SPath.hh"
#include "OpticksGenstep.h"

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "SOpticksResource.hh"

#include "CSGFoundry.h"
#include "CSGGenstep.h"
#include "CSGOptiX.h"
#include "RG.h"

#include "QSim.hh"
#include "QEvent.hh"
#include "SEvent.hh"


const char* OutDir(const Opticks& ok, const char* cfbase, const char* name)
{
    int optix_version_override = CSGOptiX::_OPTIX_VERSION(); 
    const char* out_prefix = ok.getOutPrefix(optix_version_override);   
    // out_prefix includes values of envvars OPTICKS_GEOM and OPTICKS_RELDIR when defined
    const char* default_outdir = SPath::Resolve(cfbase, name, out_prefix, DIRPATH );  
    const char* outdir = SSys::getenvvar("OPTICKS_OUTDIR", default_outdir );  

    LOG(info) 
        << " optix_version_override " << optix_version_override
        << " out_prefix [" << out_prefix << "]" 
        << " cfbase " << cfbase 
        << " default_outdir " << default_outdir
        << " outdir " << outdir
        ;

    return outdir ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv ); 
    ok.configure(); 
    ok.setRaygenMode(RG_SIMULATE) ; // override --raygenmode option 

    const char* NAME = "CSGOptiXSimulateTest" ; 
    const char* cfbase = SOpticksResource::CFBase(); 
    const char* outdir = OutDir(ok, cfbase, NAME);    
    ok.setOutDir(outdir); 
    ok.writeOutputDirScript(outdir) ; // writes CSGOptiXSimulateTest_OUTPUT_DIR.sh in PWD 
    // TODO: relocate script writing and dir mechanices into SOpticksResource or SOpticks or similar 
    ok.dumpArgv(NAME); 

    // HMM: note use of the standard OPTICKS_KEY geocache 
    const char* idpath = ok.getIdPath(); 
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
    CSGFoundry::CopyBndName(fdl, fd );  

    unsigned primIdx = 0 ; 
    const char* boundary = "Water///Pyrex" ; 
    fdl->setPrimBoundary( primIdx, boundary ); 

    std::cout << "fdl.detailPrim " << std::endl << fdl->detailPrim() ; 


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

    cx.setGensteps(&gs, 1); 
    cx.simulate();  

    const char* odir = SPath::Resolve(cfbase_local, "CSGOptiXSimulateTest", DIRPATH ); 
    
    NP* p = cx.event->getPhotons() ; 
    NP* r = cx.event->getRecords() ; 
    LOG(info) << " p " << ( p ? p->sstr() : "-" ) ; 
    LOG(info) << " r " << ( r ? r->sstr() : "-" ) ; 
    LOG(info) << " odir " << odir ; 
    if(p) p->save(odir, "p.npy"); 
    if(r) r->save(odir, "r.npy"); 

 
    cudaDeviceSynchronize(); 
    return 0 ; 
}
