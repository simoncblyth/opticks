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

    U4Material::LoadBnd(); // create G4 materials from SSim::Load bnd.npy, used by U4VolumeMaker::PV PMTSim

    SEventConfig::SetRGModeSimtrace();    // HMM: maybe discern from the simulate/simtrace/render call ? is it needed ahead of that ?
    SEventConfig::SetCompMask("genstep,simtrace"); // defaults for this should vary with the RGMode  

    //SOpticks::WriteOutputDirScript() ;   // still needed ? could be done by SEvt ctor ? 
    // writes g4cx/G4CXSimtraceTest_OUTPUT_DIR.sh in invoking directory containing
    // export G4CXSimtraceTest_OUTPUT_DIR=$TMP

    SEvt evt ;    

    Opticks::Configure(argc, argv, "--gparts_transform_offset" );  

    G4CXOpticks gx ;  
    gx.setGeometry(); 
    gx.simtrace(); 

    cudaDeviceSynchronize(); 
    evt.save();    // $DefaultOutputDir   /tmp/$USER/opticks/SProc::ExecutableName/GEOM  
 
    return 0 ; 
}
