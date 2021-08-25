/**
CSGOptiXSimulate
=================

::

     MOI=Hama CEGS=5:0:5:1000   CSGOptiXSimulate
     MOI=Hama CEGS=10:0:10:1000 CSGOptiXSimulate

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
    ok.setOutDir("$TMP/CSGOptiX/CSGOptiXSimulate");   // override --outdir option and OPTICKS_OUTDIR envvar from OpticksCfg

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
         
        std::vector<int> vcegs ; 
        SSys::getenvintvec("CEGS", vcegs, ':', "5:0:5:1000" ); 
        uint4 cegs ; 
        cegs.x = vcegs.size() > 0 ? vcegs[0] : 5  ; 
        cegs.y = vcegs.size() > 1 ? vcegs[1] : 0  ; 
        cegs.z = vcegs.size() > 2 ? vcegs[2] : 5 ; 
        cegs.w = vcegs.size() > 3 ? vcegs[3] : 1000 ; 

        int4 oce ; 
        oce.x = vcegs.size() > 4 ? vcegs[4] : 0 ; 
        oce.y = vcegs.size() > 5 ? vcegs[5] : 0 ; 
        oce.z = vcegs.size() > 6 ? vcegs[6] : 0 ; 
        oce.w = vcegs.size() > 7 ? vcegs[7] : 0 ; 

        if( oce.w > 0 )   // require 8 delimited ints to override the MOI.ce
        {
            ce.x = float(oce.x); 
            ce.y = float(oce.y); 
            ce.z = float(oce.z); 
            ce.w = float(oce.w); 
            LOG(info) << "override the MOI.ce with CEGS.ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;  
        } 
 
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
    
    //evt->savePhoton(ok.getOutDir(),  "photons.npy");   // this one gets very big 
    evt->saveGenstep(ok.getOutDir(), "genstep.npy");  

    const char* namestem = "CSGOptiXSimulate" ; 
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
