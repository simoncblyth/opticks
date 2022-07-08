/**
G4CXSimtraceTest.cc :  WIP : drawing on cx:tests/CSGOptiXSimtraceTest.cc
==========================================================================


**/

#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"
#include "SEventConfig.hh"
#include "SEvt.hh"
#include "Opticks.hh"   
#include "U4Material.hh"
#include "G4CXOpticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    U4Material::LoadBnd(); // create G4 materials from SSim::Load bnd.npy, used by U4VolumeMaker::PV PMTSim

    SEventConfig::SetRGModeSimulate();    // HMM: maybe discern from the simulate/simtrace/render call ? is it needed ahead of that ?
    SEventConfig::SetCompMask("genstep,simtrace"); 

    SOpticks::WriteOutputDirScript() ; 


    SEvt evt ;    

    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  

    G4CXOpticks gx ;  
    gx.setGeometry(); 


    sframe fr = gx.fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 



    gx.simtrace(); 

    cudaDeviceSynchronize(); 
    evt.save();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  then ALL from setRelDir
 
    return 0 ; 
}
