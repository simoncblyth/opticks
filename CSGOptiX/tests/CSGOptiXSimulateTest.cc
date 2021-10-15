/**
CSGOptiXSimulateTest
======================

Using as much as possible the CSGOptiX rendering machinery 
to do simulation. Using CSGOptiX raygen mode 1 which flips the case statement 
in the __raygen__rg program. 

The idea is to minimize code divergence between ray trace rendering and 
ray trace enabled simulation. Because after all the point of the rendering 
is to be able to see the exact same geometry that the simulation is using.

::

     MOI=Hama CEGS=5:0:5:1000   CSGOptiXSimulateTest
     MOI=Hama CEGS=10:0:10:1000 CSGOptiXSimulateTest

**/

#include <cuda_runtime.h>
#include <algorithm>
#include <iterator>

#include "SSys.hh"
#include "SPath.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "CSGOptiXSimulate.h"

#include "QSim.hh"
#include "QEvent.hh"



int main(int argc, char** argv)
{
    for(int i=0 ; i < argc ; i++ ) std::cout << argv[i] << std::endl; 

    OPTICKS_LOG(argc, argv); 

    const char* cxs = SSys::getenvvar("CXS", "0" ); 
    int create_dirs = 0 ;  
    const char* default_outdir = SPath::Resolve("$TMP/CSGOptiX/CSGOptiXSimulateTest",  cxs, create_dirs );  
    SSys::setenvvar("OPTICKS_OUTDIR", default_outdir , false );  // change default, but allow override by evar

    Opticks ok(argc, argv); 
    ok.configure(); 
    ok.setRaygenMode(1) ; // override --raygenmode option 

    const char* outdir = ok.getOutDir(); 
    LOG(info) << " cxs " << cxs << " outdir " << outdir ; 

    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );
    const char* topline = SSys::getenvvar("TOPLINE", "CSGOptiXRender") ; 
    const char* botline = SSys::getenvvar("BOTLINE", nullptr ) ; 


    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    fd->upload(); 

    // GPU physics uploads : boundary+scintillation textures, property+randomState arrays    
    QSim<float>::UploadComponents(fd->icdf, fd->bnd ); 

    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    CSGOptiX cx(&ok, fd); 
    cx.setTop(top); 

    if( cx.raygenmode == 0 )
    {
        LOG(fatal) << " WRONG EXECUTABLE FOR CSGOptiX::render cx.raygenmode " << cx.raygenmode ; 
        assert(0); 
    }

    const char* moi = SSys::getenvvar("MOI", "sWorld:0:0");  
    const NP* gs = nullptr ; 
    if( strcmp(moi, "FAKE") == 0 )
    { 
        std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
        gs = QEvent::MakeCountGensteps(photon_counts_per_genstep) ;
    }
    else
    {
        float4 ce = make_float4(0.f, 0.f, 0.f, 1000.f ); 
        int midx, mord, iidx ; 
        fd->parseMOI(midx, mord, iidx, moi );  
        LOG(info) << " moi " << moi << " midx " << midx << " mord " << mord << " iidx " << iidx ;   
        int rc = fd->getCenterExtent(ce, midx, mord, iidx) ;
        LOG(info) << " rc " << rc << " MOI.ce (" 
              << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;           

        uint4 cegs ; 
        CSGOptiXSimulate::ParseCEGS(cegs, ce ); 

        gs = QEvent::MakeCenterExtentGensteps(ce, cegs); 
        cx.setCE(ce); 
        cx.setCEGS(cegs); 
    }

    cx.setGensteps(gs); 

    double dt = cx.simulate();  
    LOG(info) << " dt " << dt ;

    QSim<float>* sim = cx.sim ; 
    QEvent* evt = cx.evt ; 
    
    //evt->savePhoton(ok.getOutDir(),  "photons.npy");   // this one gets very big 
    evt->saveGenstep(ok.getOutDir(), "genstep.npy");  

    const char* namestem = "CSGOptiXSimulateTest" ; 
    const char* ext = ".jpg" ; 
    int index = -1 ;  
    const char* outpath = ok.getOutPath(namestem, ext, index ); 
    LOG(error) << " outpath " << outpath ; 

    std::string bottom_line = CSGOptiX::Annotation(dt, botline ); 
    cx.snap(outpath, bottom_line.c_str(), topline  );   
    cx.writeFramePhoton(ok.getOutDir(), "fphoton.npy" ); 


    cudaDeviceSynchronize(); 
    return 0 ; 
}
