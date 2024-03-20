/**

~/o/sysrap/tests/SCUDA_Mesh_test.sh

**/

#include "SCUDA_Mesh.h"
#include "SOPTIX.h"

int main()
{
    SMesh* m = SMesh::Load("$SCENE_FOLD/scene/mesh_grup/3" ); 
    std::cout << m->desc() ; 

    SOPTIX ox ; 
    std::cout << ox.desc() ; 

    SCUDA_Mesh* mesh = new SCUDA_Mesh(m) ; 
    std::cout << mesh->desc() ; 



    //CUdeviceptr p_vtx = (CUdeviceptr)(uintptr_t) mesh->vtx.ptr ; 
    CUdeviceptr p_vtx = mesh->vtx.pointer() ; 
    unsigned numVertices = mesh->vtx.num_item/3 ; 
    unsigned vertexStrideInBytes = 0 ; 

    CUdeviceptr indexBuffer = mesh->idx.pointer() ; 
    unsigned numIndexTriplets = mesh->idx.num_item/3 ; 

    OptixBuildInputType type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;
    OptixVertexFormat vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3 ; 
    OptixIndicesFormat indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;  
    unsigned indexStrideInBytes = 0 ; 
    CUdeviceptr preTransform = 0 ; 
    OptixTransformFormat transformFormat = OPTIX_TRANSFORM_FORMAT_NONE ;  

    OptixBuildInput buildInput = {} ;
    buildInput.type = type ; 
    OptixBuildInputTriangleArray& triangleArray = buildInput.triangleArray ;

    unsigned flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING ; 
    const unsigned* flags = &flag ;  
    unsigned numSbtRecords = 1 ; 
    CUdeviceptr sbtIndexOffsetBuffer = 0 ; 
    unsigned sbtIndexOffsetSizeInBytes = 0 ; 
    unsigned sbtIndexOffsetStrideInBytes = 0 ; 
    unsigned primitiveIndexOffset = 0 ; 

    triangleArray.vertexBuffers = &p_vtx ;
    triangleArray.numVertices = numVertices ;  
    triangleArray.vertexFormat = vertexFormat ; 
    triangleArray.vertexStrideInBytes = vertexStrideInBytes ; 
    triangleArray.indexBuffer = indexBuffer ; 
    triangleArray.numIndexTriplets = numIndexTriplets ; 
    triangleArray.indexFormat = indexFormat ;     
    triangleArray.indexStrideInBytes = indexStrideInBytes ; 
    triangleArray.preTransform = preTransform ; 
    triangleArray.flags = flags ; 
    triangleArray.numSbtRecords = numSbtRecords ;    
    triangleArray.sbtIndexOffsetBuffer = sbtIndexOffsetBuffer ; 
    triangleArray.sbtIndexOffsetSizeInBytes = sbtIndexOffsetSizeInBytes ; 
    triangleArray.sbtIndexOffsetStrideInBytes = sbtIndexOffsetStrideInBytes ;
    triangleArray.primitiveIndexOffset = primitiveIndexOffset ; 
    triangleArray.transformFormat = transformFormat ;     

 

}
