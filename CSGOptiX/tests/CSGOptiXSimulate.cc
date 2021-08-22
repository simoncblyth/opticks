/**
CSGOptiXSimulate
=================

**/

#include <algorithm>
#include <iterator>

#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
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

    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );
    const char* outdir = SSys::getenvvar("OUTDIR", "$TMP/CSGOptiX/CSGOptiXSimulate" );

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

    double dt = cx.simulate();  
    LOG(info) << " dt " << dt ;


    QSim<float>* sim = cx.sim ; 
    QEvent* evt = cx.evt ; 
    
    /*
    std::vector<quad4> photon ; 
    evt->downloadPhoton(photon); 
    LOG(info) << " downloadPhoton photon.size " << photon.size() ; 
    sim->dump_photon( photon.data(), photon.size(), "f0,f1,f2,f3" );    // TODO: move dumping into QEvent
    */

    evt->savePhoton(outdir, "photons.npy");  


    cudaDeviceSynchronize(); 

    return 0 ; 
}
