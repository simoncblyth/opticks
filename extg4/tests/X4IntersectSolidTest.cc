/**
X4IntersectSolidTest
======================

Used from script extg4/xxs.sh 

**/
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "SPath.hh"

#include "X4Intersect.hh"
#include "X4Mesh.hh"

#include "X4_GetSolid.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //const char* geom_default = "pmt_solid" ; // not appropriate for default to require j/PMTSim 
    const char* geom_default = "Orb" ; 
    const char* geom = SSys::getenvvar("GEOM", geom_default );  

    std::vector<std::string> names ; 
    SStr::Split(geom,',',names);  
    LOG(info) << " geom " << geom << " names.size " << names.size() ; 


    std::stringstream ss ; 
    ss << "creator:X4IntersectSolidTest" << std::endl ; 
    ss << "geom:" << geom << std::endl ; 
#ifdef WITH_PMTSIM
    ss << "info:WITH_PMTSIM " << std::endl ; 
#else
    ss << "info:noPMTSIM " << std::endl ; 
#endif
    std::string meta = ss.str(); 
    LOG(info) << meta ; 


    const char* base = "$TMP/extg4/X4IntersectSolidTest" ; 

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const std::string& name_ = names[i]; 
        const char* name = name_.c_str() ; 
        const G4VSolid* solid = X4_GetSolid(name); 
        if( solid == nullptr ) LOG(fatal) << "failed to X4_GetSolid for name " << name ; 
        assert( solid );   

        X4Intersect::Scan(solid, name, base, meta ); 

        /**
        its simpler to do this in another executable : X4MeshTest 

        int create_dirs = 1 ; // 1:filepath 
        const char* meshpath = SPath::Resolve(base, name, "X4Mesh", "mesh.gltf", create_dirs ); 
        X4Mesh::Save(solid, meshpath ); 

        **/

    }
    return 0 ; 
}

