/**
CSGOptiXSimulate
=================

::

     MOI=Hama CEG=5:0:5:1000 CSGOptiXSimulate
     MOI=Hama CEG=10:0:10:1000 CSGOptiXSimulate

**/

#include <cuda_runtime.h>
#include <algorithm>
#include <iterator>

#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"

#include "QSim.hh"
#include "QEvent.hh"


int main(int argc, char** argv)
{
    for(int i=0 ; i < argc ; i++ ) std::cout << argv[i] << std::endl; 

    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 
    ok.setRaygenMode(1) ; // override --raygenmode option 
    ok.setOutDir("$TMP/CSGOptiX");   // override --outdir option 

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
        LOG(info) << " rc " << rc << " ce " << ce.x << " " << ce.y << " " << ce.z << " " << ce.w ;  
         
        std::vector<int> vcegs ; 
        SSys::getenvintvec("CEGS", vcegs, ':', "5:0:5:1000" ); 
        uint4 cegs ; 
        cegs.x = vcegs.size() > 0 ? vcegs[0] : 5  ; 
        cegs.y = vcegs.size() > 1 ? vcegs[1] : 0  ; 
        cegs.z = vcegs.size() > 2 ? vcegs[2] : 5 ; 
        cegs.w = vcegs.size() > 3 ? vcegs[3] : 1000 ; 

        LOG(info) 
            << " CEGS nx:ny:nz:photons_per_genstep " << cegs.x << ":" << cegs.y << ":" << cegs.z << ":" << cegs.w 
            ;   

        gs = QEvent::MakeCenterExtentGensteps(ce, cegs); 
        cx.setCE(ce); 
        cx.setCEGS(cegs); 
    }

    cx.setGensteps(gs); 

    double dt = cx.simulate();  
    LOG(info) << " dt " << dt ;

    QSim<float>* sim = cx.sim ; 
    QEvent* evt = cx.evt ; 
    
    evt->savePhoton(ok.getOutDir(),  "photons.npy");  
    evt->saveGenstep(ok.getOutDir(), "genstep.npy");  

    const char* namestem = "CSGOptiXSimulate" ; 
    const char* ext = ".jpg" ; 
    int index = -1 ;  
    const char* outpath = ok.getOutPath(namestem, ext, index ); 
    LOG(error) << " outpath " << outpath ; 

    std::string bottom_line = CSGOptiX::Annotation(dt, botline ); 
    cx.snap(outpath, bottom_line.c_str(), topline  );   


    cudaDeviceSynchronize(); 
    return 0 ; 
}
