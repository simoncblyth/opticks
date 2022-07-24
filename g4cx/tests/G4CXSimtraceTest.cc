/**
G4CXSimtraceTest.cc :  based on cx:tests/CSGOptiXSimtraceTest.cc
==========================================================================

TODO : rename "simtrace" to "trace" everywhere for clarity and abbrev convenience

::

    s:simulate  => gxs.sh cxs.sh
    t:trace     => gxt.sh cxt.sh
    r:render    => gxr.sh cxr.sh 


**/

#include <cuda_runtime.h>

#include "OPTICKS_LOG.hh"
#include "SOpticks.hh"


#include "SEventConfig.hh"
#include "SEvt.hh"
#include "Opticks.hh"   
#include "U4Material.hh"
#include "G4CXOpticks.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks::Configure(argc, argv, "--gparts_transform_offset --allownokey" );  

    // U4Material::LoadBnd(); 
    // TODO: work out when LoadBnd is needed and relocate inside G4CXOpticks::setGeometry
    // create G4 materials from SSim::Load bnd.npy, used by U4VolumeMaker::PV PMTSim
    // HMM: probably dont want to do this when running from GDML

    SEventConfig::SetRGModeSimtrace();    // HMM: maybe discern from the simulate/simtrace/render call ? is it needed ahead of that ?
    SEventConfig::SetCompMask("genstep,simtrace"); // defaults for this should vary with the RGMode  

    SEvt evt ;    
    evt.setReldir("ALL"); 

    G4CXOpticks gx ;  
    gx.setGeometry(); 
    gx.simtrace(); 

    cudaDeviceSynchronize(); 
    evt.save();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  
 
    return 0 ; 
}
