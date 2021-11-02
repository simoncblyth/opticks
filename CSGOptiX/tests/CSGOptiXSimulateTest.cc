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

     MOI=Hama CXS_CEGS=5:0:5:1000   CSGOptiXSimulateTest
     MOI=Hama CXS_CEGS=10:0:10:1000 CSGOptiXSimulateTest

**/

#include <cuda_runtime.h>
#include <algorithm>
#include <iterator>

#include "scuda.h"
#include "sqat4.h"
#include "stran.h"

#include "SSys.hh"
#include "SPath.hh"

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "CSGFoundry.h"
#include "CSGOptiX.h"

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

    const char* argforced = "--allownokey" ; 
    Opticks ok(argc, argv, argforced); 
    ok.configure(); 
    ok.setRaygenMode(1) ; // override --raygenmode option 

    const char* outdir = ok.getOutDir(); 
    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );  // CFBASE dir must contain CSGFoundry subdir 
    const char* botline = SSys::getenvvar("BOTLINE", nullptr ) ; 

    const char* moi = SSys::getenvvar("MOI", "sWorld:0:0");  
    float gridscale = SSys::getenvfloat("GRIDSCALE", 1.0 ); 
    const char* topline = SSys::getenvvar("TOPLINE", "CSGOptiXRender") ; 

    LOG(info) << " cxs " << cxs << " outdir " << outdir << " gridscale " << gridscale ; 

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

        qat4 qt ; qt.init();  // initize ti identity 
        int rc = fd->getCenterExtent(ce, midx, mord, iidx, &qt ) ;
        LOG(info) << " rc " << rc << " MOI.ce (" 
              << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;           

        LOG(info) << std::endl << "qt" << qt ; 
        Tran<double>* geotran = Tran<double>::ConvertToTran( &qt );  // houses transform and inverse


        std::vector<int> cegs ; 
        SSys::getenvintvec("CXS_CEGS", cegs, ':', "5:0:5:1000" ); 
        // expect 4 or 7 ints delimited by colon nx:ny:nz:num_pho OR nx:px:ny:py:nz:py:num_pho 

        QEvent::StandardizeCEGS(ce, cegs, gridscale ); 
        assert( cegs.size() == 7 ); 

        std::vector<int> override_ce ; 
        SSys::getenvintvec("CXS_OVERRIDE_CE",  override_ce, ':', "0:0:0:0" ); 

        if( override_ce.size() == 4 && override_ce[3] > 0 )
        {
            ce.x = float(override_ce[0]); 
            ce.y = float(override_ce[1]); 
            ce.z = float(override_ce[2]); 
            ce.w = float(override_ce[3]); 
            LOG(info) << "override the MOI.ce with CXS_OVERRIDE_CE (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;  
        } 


        gs = QEvent::MakeCenterExtentGensteps(ce, cegs, gridscale, geotran ); 
        cx.setCE(ce); 
        cx.setCEGS(cegs); 
        cx.setMetaTran(geotran); 

        //cx.setNear(0.1); // TODO: not getting 0.1., investigate 
    }

    cx.setGensteps(gs); 

    double dt = cx.simulate();  
    LOG(info) << " dt " << dt ;

    //QSim<float>* sim = cx.sim ;
 
    QEvent* evt = cx.evt ; 
    evt->setMeta( fd->meta ); 
    evt->savePhoton( outdir, "photons.npy");   // this one can get very big 
    evt->saveGenstep(outdir, "genstep.npy");  
    evt->saveMeta(   outdir, "fdmeta.txt" ); 

    const char* namestem = "CSGOptiXSimulateTest" ; 
    const char* ext = ".jpg" ; 
    int index = -1 ;  
    const char* outpath = ok.getOutPath(namestem, ext, index ); 
    LOG(error) << " outpath " << outpath ; 

    std::string bottom_line = CSGOptiX::Annotation(dt, botline ); 
    cx.snap(outpath, bottom_line.c_str(), topline  );   
    cx.writeFramePhoton(outdir, "fphoton.npy" );   // as only 1 possible frame photon per-pixel the size never gets very big 
    cx.savePeta(outdir, "peta.npy");   
    cx.saveMetaTran(outdir, "metatran.npy"); 


    cudaDeviceSynchronize(); 
    return 0 ; 
}
