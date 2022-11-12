#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "X4Solid.hh"
#include "X4Mesh.hh"
#include "X4_Get.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* geom_default = "hmskSolidMask" ; 
    const char* geom = SSys::getenvvar("X4MeshTest_GEOM", geom_default ); 

    std::string meta ; 
    const G4VSolid* solid = X4_Get::GetSolid(geom, &meta); 

    bool has_meta = !meta.empty() ; 
    LOG_IF(info, has_meta) << "meta:" << std::endl << meta ; 

    LOG_IF(fatal, solid == nullptr) << "failed to X4_GetSolid for geom " << geom ; 
    if(!solid) return 1 ; 

    const char* base = "$TMP/extg4/X4MeshTest" ; 
    const char* meshpath = SPath::Resolve(base, geom, "X4Mesh", "mesh.gltf", FILEPATH ); 
    LOG(info) << " save to " << meshpath ; 

    X4Mesh::Save(solid, meshpath ); 

    return 0 ; 
}
