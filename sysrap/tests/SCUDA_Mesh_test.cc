/**

~/o/sysrap/tests/SCUDA_Mesh_test.sh


optix7.5 p28
   Mixing build input types in a single geometry-AS is not allowed
   [will need separated triangulated GAS for guidetube]

**/


#include "scuda.h"



#include "SCUDA_Mesh.h"
#include "SOPTIX_Context.h"
#include "SOPTIX_Desc.h"

int main()
{
    SMesh* m = SMesh::Load("$SCENE_FOLD/scene/mesh_grup/3" ); 
    std::cout << m->desc() ; 

    //SOPTIX_Context ox ; 
    //std::cout << ox.desc() << std::endl ; 

    SCUDA_Mesh* _mesh = new SCUDA_Mesh(m) ; 
    std::cout << _mesh->desc() ; 


    return 0 ; 
}
