/**
X4IntersectTest
=================

Used from script x4.sh 

**/
#include "G4Orb.hh"
#include "OPTICKS_LOG.hh"
#include "SSys.hh"

#include "X4Intersect.hh"
#include "X4GeometryMaker.hh"

#ifdef WITH_PMTSIM
#include "PMTSim.hh"
#endif

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* name_default = "pmt_solid" ; 
    const char* name = SSys::getenvvar("GEOM", name_default );  

    std::stringstream ss ; 
    ss << "creator:X4IntersectTest" << std::endl ; 
    ss << "name:" << name << std::endl ; 
#ifdef WITH_PMTSIM
    ss << "info:WITH_PMTSIM " << std::endl ; 
#else
    ss << "info:noPMTSIM " << std::endl ; 
#endif
    std::string meta = ss.str(); 
    LOG(info) << meta ; 

    const G4VSolid* solid = nullptr ; 

    if( strcmp(name, "orb") == 0 )
    {
        solid = new G4Orb( name, 100. );
    }
    else if(X4GeometryMaker::CanMake(name))
    {
        solid = X4GeometryMaker::Make(name); 
    }
    else
    {
#ifdef WITH_PMTSIM
        solid = PMTSim::GetSolid(name); 
#endif
    }
    X4Intersect::Scan(solid, name, "$TMP/extg4/X4IntersectTest", meta ); 
    return 0 ; 
}

