#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "X4Solid.hh"
#include "X4Mesh.hh"
#include "X4_GetSolid.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* geom_default = "hmsk_solidMask" ; 
    const char* geom = SSys::getenvvar("GEOM", geom_default ); 

    std::string meta ; 
    const G4VSolid* solid = X4_GetSolid(geom, meta ); 
    if(!meta.empty()) LOG(info) << "meta:" << std::endl << meta ; 

    if( solid == nullptr ) LOG(fatal) << "failed to X4_GetSolid for geom " << geom ; 
    if(!solid) return 1 ; 

    const char* base = "$TMP/extg4/X4MeshTest" ; 
    int create_dirs = 1 ; // 1:filepath 
    const char* meshpath = SPath::Resolve(base, geom, "X4Mesh", "mesh.gltf", create_dirs ); 
    LOG(info) << " save to " << meshpath ; 

    X4Mesh::Save(solid, meshpath ); 

    return 0 ; 
}
