/**
CSGOptiXSimtraceTest
======================

Using as much as possible the CSGOptiX rendering machinery 
to do simulation. Using CSGOptiX raygen mode 1 which flips the case statement 
in the __raygen__rg program. 

The idea is to minimize code divergence between ray trace rendering and 
ray trace enabled simulation. Because after all the point of the rendering 
is to be able to see the exact same geometry that the simulation is using.

::

     MOI=Hama CEGS=5:0:5:1000   CSGOptiXSimtraceTest
     MOI=Hama CEGS=10:0:10:1000 CSGOptiXSimtraceTest

TODO: 
    find way to get sdf distances for all intersects GPU side 
    using GPU side buffer of positions 

    just need to run the distance function for all points and with 
    the appropriate CSGNode root and num_node

    does not need OptiX, just CUDA (or it can be done CPU side also)

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

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "CSGFoundry.h"
#include "CSGGenstep.h"
#include "CSGOptiX.h"
#include "RG.h"

#include "QSim.hh"
#include "QEvent.hh"
#include "SEvent.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv ); 
    ok.configure(); 

    int raygenmode = RG_SIMTRACE ;  
    ok.setRaygenMode(raygenmode) ; // override --raygenmode option 

    int optix_version_override = CSGOptiX::_OPTIX_VERSION(); 
    const char* out_prefix = ok.getOutPrefix(optix_version_override);   
    // out_prefix includes values of envvars OPTICKS_GEOM and OPTICKS_RELDIR when defined
    LOG(info) 
        << " optix_version_override " << optix_version_override
        << " out_prefix [" << out_prefix << "]" 
        ;

    const char* cfbase = ok.getFoundryBase("CFBASE");  // envvar CFBASE can override 
    const char* default_outdir = SPath::Resolve(cfbase, "CSGOptiXSimtraceTest", out_prefix, DIRPATH );  
    const char* outdir = SSys::getenvvar("OPTICKS_OUTDIR", default_outdir );  

    ok.setOutDir(outdir); 
    ok.writeOutputDirScript(outdir) ; // writes CSGOptiXSimtraceTest_OUTPUT_DIR.sh in PWD 

    const char* outdir2 = ok.getOutDir(); 
    assert( strcmp(outdir2, outdir) == 0 ); 

    LOG(info) 
        << " cfbase " << cfbase 
        << " default_outdir " << default_outdir
        << " outdir " << outdir
        ; 

    ok.dumpArgv("CSGOptiXSimtraceTest"); 

    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* topline = SSys::getenvvar("TOPLINE", "CSGOptiXRender") ; 
    const char* botline = SSys::getenvvar("BOTLINE", nullptr ) ; 

    const char* idpath = ok.getIdPath(); 
    const char* rindexpath = SPath::Resolve(idpath, "GScintillatorLib/LS_ori/RINDEX.npy", 0 );  
    
    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    if(fd->hasMeta()) LOG(info) << "fd.meta\n" << fd->meta ; 

    fd->upload(); 

    // GPU physics uploads : boundary+scintillation textures, property+randomState arrays    
    QSim<float>::UploadComponents(fd->icdf, fd->bnd, fd->optical, rindexpath ); 

    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    CSGOptiX cx(&ok, fd); 

    // create center-extent gensteps 
    CSGGenstep* gsm = fd->genstep ;    // THIS IS THE GENSTEP MAKER : NOT THE GS THEMSELVES 
    const char* moi = SSys::getenvvar("MOI", "sWorld:0:0");  
    bool ce_offset = SSys::getenvint("CE_OFFSET", 0) > 0 ; 
    bool ce_scale = SSys::getenvint("CE_SCALE", 0) > 0 ;   
    // TODO: eliminate the need for this switches by standardizing on model2world transforms

    gsm->create(moi, ce_offset, ce_scale ); // SEvent::MakeCenterExtentGensteps

    NP* gs = gsm->gs ; 
    gs->set_meta<std::string>("TOP", top ); 
    gs->set_meta<std::string>("TOPLINE", topline ); 
    gs->set_meta<std::string>("BOTLINE", botline ); 

    cx.setComposition(gsm->ce, gsm->m2w, gsm->w2m ); 
    cx.setCEGS(gsm->cegs);   // sets peta metadata
    cx.setMetaTran(gsm->geotran); 
    cx.setGensteps(gs); 

    cx.simulate();    // actually doing "simtrace" thanks to  
    cx.snapSimulateTest(outdir, botline, topline ); 
 
    cudaDeviceSynchronize(); 
    return 0 ; 
}
