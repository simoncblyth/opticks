/**
X4_Get.hh  (formerly X4_GetSolid.hh but want to generalize to volume getting)
================================================================================

::

    epsilon:tests blyth$ grep -l X4_Get.hh *.* | grep -v X4_Get.hh 
    X4IntersectSolidTest.cc
    X4MeshTest.cc
    X4MeshTest0.cc
    X4SimtraceTest.cc
    epsilon:tests blyth$ 


Depending on the PMTSim CMake target is needed to access the JUNO Specific j/PMTSim solids 
and volumes. Only a few separately listed extg4 tests depend on PMTSim.
Access to these solids and volumes requires separate building of PMTSim 
and installation into the Opticks CMAKE_PREFIX_PATH 
This can be done with::

    cd ~/j/PMTSim   # jps  
    om 

**/

#include "SStr.hh"
#include "X4SolidMaker.hh"
#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

namespace X4_Get 
{
    inline const G4VSolid* GetSolid(const char* geom, std::string* meta=nullptr ); 
}

namespace X4_Get 
{
    inline const G4VSolid* GetSolid(const char* geom, std::string* meta )
    {
        const char* geom_upto_first_underscore = SStr::HeadFirst(geom, '_');  

        const G4VSolid* solid = nullptr ; 
        if(X4SolidMaker::CanMake(geom_upto_first_underscore ))
        {
            solid = X4SolidMaker::Make(geom_upto_first_underscore , meta); 
        }
        else
        {
    #ifdef WITH_PMTSIM
            std::cout << "extg4/tests/X4_Get.hh : X4_Get::GetSolid : WITH_PMTSIM invoking PMTSim::GetSolid(\"" << geom << "\") " << std::endl ; 
            solid = PMTSim::GetSolid(geom);   // need geom for option passing following "__" 
    #else
            std::cout << "extg4/tests/X4_Get.hh : X4_Get::GetSolid :  not WITH_PMTSIM " << geom << std::endl ;  
    #endif
        }
        return solid ; 
    }
}




