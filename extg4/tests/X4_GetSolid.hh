/**
X4_GetSolid.hh
================

Depending on the PMTSim CMake target is needed to access the JUNO Specific j/PMTSim solids 
and volumes. Only a few separately listed extg4 tests depend on PMTSim.
Access to these solids and volumes requires separate building of PMTSim 
and installation into the Opticks CMAKE_PREFIX_PATH 
This can be done with::

    cd ~/j/PMTSim   # jps  
    om 

**/

#include "X4SolidMaker.hh"
#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

const G4VSolid* X4_GetSolid(const char* name)
{
    const G4VSolid* solid = nullptr ; 
    if(X4SolidMaker::CanMake(name))
    {
        solid = X4SolidMaker::Make(name); 
    }
    else
    {
#ifdef WITH_PMTSIM
        solid = PMTSim::GetSolid(name); 
#else
        std::cout << " not WITH_PMTSIM " << name << std::endl;  
#endif
    }
    return solid ; 
}


