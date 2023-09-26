#include "ssys.h"
#include "G4VSolid.hh"
#include "U4Mesh.h"
#include "PMTSim.hh"

int main()
{
    const char* GEOM = ssys::getenvvar("GEOM","NoGEOM") ; 
    G4VSolid* solid = PMTSim::GetSolid(GEOM);
    std::string desc = PMTSim::Desc(GEOM); 
    if(solid == nullptr) return 1 ; 
    G4cout << *solid << std::endl ; 

    NPFold* fold = U4Mesh::Serialize(solid) ; 
    fold->set_meta<std::string>("GEOM",GEOM);      
    fold->set_meta<std::string>("desc",desc);      
    fold->save("$FOLD"); 
    return 0 ; 
};

