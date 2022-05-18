/**
CSGOptiXSimulateTest
======================

Canonically used by cxsim.sh combining CFBASE_LOCAL simple test geometry (eg GeoChain) 
with standard CFBASE basis geometry  

BUT: the basis "geometry" (the first CSGFoundry loaded) is not actually needed, 
it is the basis QSim components that are required. 

Having to load the full CSGFoundry geometry and then not use it just seems wrong. 
 
Better for QSim to manage its own components that are persisted within a 
subdirectory(or sibling) of the CSGFoundry dir.  
Then can do QSim::Load and pass QSim instance to CSGOptiX 

That better reflects the intention for a rather loose relationship between CSGFoundry and QSim. 

This means CSG_GGeo needs to convert the traditional GGeo into both CSGFoundry and QSim
instances/directories rather than the QSim components living as foreign NP inside CSGFoundry. 

**/

#include <cuda_runtime.h>

#include "scuda.h"
#include "sqat4.h"
#include "scarrier.h"
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

#ifdef WITH_SGLM
// SGLM replaces Composition, preventing the need for Opticks instance
#else
#include "Opticks.hh"
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

#ifdef WITH_SGLM
#else
    Opticks::Configure(argc, argv );  
#endif
    const char* cfbase = SOpticksResource::CFBase(); 
    const char* outfold = SEventConfig::OutFold();  
    LOG(info) << " outfold [" << outfold << "]"  ; 

    SOpticks::WriteOutputDirScript(outfold) ; // writes CSGOptiXSimulateTest_OUTPUT_DIR.sh in PWD 

    const char* idpath = SOpticksResource::IDPath();  // HMM: using OPTICKS_KEY geocache, not CSGFoundry
    const char* rindexpath = SPath::Resolve(idpath, "GScintillatorLib/LS_ori/RINDEX.npy", 0 );  

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    if(fd->hasMeta()) LOG(info) << "fd.meta\n" << fd->meta ; 
    LOG(info) << fd->descComp(); 

    // GPU physics uploads : boundary+scintillation textures, property+randomState arrays    
    QSim::UploadComponents(fd->icdf, fd->bnd, fd->optical, rindexpath ); 

    QSim* sim = new QSim ; 
    LOG(info) << sim->desc(); 
    // TODO:QSim::Load, QSim::upload 


    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    const char* cfbase_local = SSys::getenvvar("CFBASE_LOCAL") ; 
    assert(cfbase_local) ; 

    LOG(fatal) << "MIXING CSGFoundry combining basis cfbase with cfbase_local "; 
    std::cout << std::setw(20) << "cfbase" << ":" << cfbase << std::endl ; 
    std::cout << std::setw(20) << "cfbase_local" << ":" << cfbase_local  << std::endl ; 

    CSGFoundry* fdl = CSGFoundry::Load(cfbase_local, "CSGFoundry") ; 

    CSGOptiX* cx = CSGOptiX::Create(fdl); 

    cx->setGenstep(SEvent::MakeCarrierGensteps());     
    cx->simulate();  

    cudaDeviceSynchronize(); 

    const char* odir = SPath::Resolve(cfbase_local, "CSGOptiXSimulateTest", DIRPATH ); 
    LOG(info) << " odir " << odir ; 
    cx->event->save(odir); 
 
    return 0 ; 
}
