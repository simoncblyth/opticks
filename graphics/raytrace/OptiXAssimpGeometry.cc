#include "OptiXAssimpGeometry.hh"
#include "AssimpNode.hh"
#include "OptiXProgram.hh"

#include <string.h>
#include <stdlib.h>
#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>

#include <optixu/optixu_vector_types.h>


OptiXAssimpGeometry::~OptiXAssimpGeometry()
{
}

OptiXAssimpGeometry::OptiXAssimpGeometry(const char* path)
           : 
           AssimpGeometry(path),
           m_context(NULL),
           m_program(NULL),
           m_material(NULL),
           m_maxdepth(4)
{
}

void OptiXAssimpGeometry::setContext(optix::Context& context)
{
    m_context = context ;   
}
void OptiXAssimpGeometry::setProgram(OptiXProgram* program)
{
    m_program = program ;   
}
void OptiXAssimpGeometry::setMaterial(optix::Material material)
{
    m_material = material ;   
}
void OptiXAssimpGeometry::setGeometryGroup(optix::GeometryGroup gg)
{
    m_geometry_group = gg ; 
}
void OptiXAssimpGeometry::setMaxDepth(unsigned int  maxdepth)
{
    m_maxdepth = maxdepth ;
}


optix::GeometryGroup OptiXAssimpGeometry::getGeometryGroup()
{
    return m_geometry_group ; 
}
optix::Material OptiXAssimpGeometry::getMaterial()
{
    return m_material ; 
}
unsigned int OptiXAssimpGeometry::getMaxDepth()
{
    return m_maxdepth ; 
}




void OptiXAssimpGeometry::convert(const char* query)
{
    //
    //  #. select AssimpNode based on query string
    //  #. traverse the AssimpNode converting aiMesh into optix::GeometryInstance
    //     collected into m_gis 
    //

    unsigned int nai = select(query);

    for(unsigned int i = 0; i < m_aiscene->mNumMaterials; i++)
    {
        optix::Material material = convertMaterial(m_aiscene->mMaterials[i]);
        m_materials.push_back(material);
    }


    m_gis.clear();

    if(nai == 0){
        printf("OptiXAssimpGeometry::convert WARNING query \"%s\" failed to find any nodes : converting root  \n", query );
        AssimpNode* root = getRoot() ;
        traverseNode(root, 0);
    } 
    else
    {
        for(unsigned int i=0 ; i < getNumSelected() ; i++ )
        {
            AssimpNode* node = getSelectedNode(i) ;
            traverseNode(node, 0);
        } 
    } 

    printf("OptiXAssimpGeometry::convert  query %s finds %d nodes with %lu gi \n", query, nai, m_gis.size() );

    // fig2:  single gg containing many gi 
    m_geometry_group->setChildCount(m_gis.size());
    for(unsigned int i=0 ; i <m_gis.size() ; i++) m_geometry_group->setChild(i, m_gis[i]);

}


void OptiXAssimpGeometry::setupAcceleration()
{
   // huh : there are currently lots of separate buffers for each gi 

    optix::Acceleration acceleration = m_context->createAcceleration("Sbvh", "Bvh");
    acceleration->setProperty( "vertex_buffer_name", "vertexBuffer" );
    acceleration->setProperty( "index_buffer_name", "indexBuffer" );

    m_geometry_group->setAcceleration( acceleration );

    acceleration->markDirty();

    m_context["top_object"]->set(m_geometry_group);
}




void OptiXAssimpGeometry::traverseNode(AssimpNode* node, unsigned int depth)
{
   //
   //  Recursive traverse of the AssimpNode converting aiMesh into optix::GeometryInstance 
   //  and collecting into m_gis 
   //
   //  NB AssimpNode meshes have been copied from locals and globally positioned by AssimpTree
   // 
   //  TODO: 
   //      try out instanced geometry and transforms, could then act on 
   //      the small number of local coordinate meshes 
   //             optix::Geometry geometry = convertGeometry(m_aiscene->mMeshes[i]);
   //             unsigned int meshIndex = node->getMeshIndexRaw(i);  // was used with local mesh handling 
   //
   //      potential large reduction in GPU memory usage 
   //
   //

    if(depth < m_maxdepth )
    {
        for(unsigned int i = 0; i < node->getNumMeshes(); i++)
        {   
            aiMesh* mesh = node->getMesh(i);   // these are copied and globally positioned meshes 

            optix::Geometry geometry = convertGeometry(mesh) ;  

            //std::vector<optix::Material>::iterator mit = m_materials.begin() + mesh->mMaterialIndex ;

            //optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, mit, mit+1  ); // single material 

            optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, &m_material, &m_material+1  ); // single material 

            m_gis.push_back(gi);
        }   
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverseNode(node->getChild(i), depth + 1);
}


optix::Material OptiXAssimpGeometry::convertMaterial(aiMaterial* ai_material)
{
    /*
    TODO:
        get assimp to access wavelength dependant material properties
        and feed them through into the material program 
        * by code gen of tables ? referencing a buffer of structs ?

        tis better to defer material association to the loader, for flexibility 
    */

    optix::Material material = m_context->createMaterial();
    const char* filename = "material1.cu" ; 
    const char* fname = "closest_hit_radiance" ; 
    optix::Program  program = m_program->createProgram(filename, fname);
    material->setClosestHitProgram(0, program);
    //material["Kd"]->setFloat( 0.7f, 0.7f, 0.7f);   // not used for normal shader in material1.cu 
    return material ; 
}


optix::Geometry OptiXAssimpGeometry::convertGeometry(aiMesh* mesh)
{
   //
   //    optix::Geometry created and populated with data from aiMesh
   //
   //    #. TriangleMesh.cu intersection and bbox programs compiled 
   //       and attached to the geometry, 
   //
   //    #. buffers/variables created and populated/set based on the aiMesh
   //       are inputs to intersection and bbox programs
   //  
   //       * vertexBuffer
   //       * normalBuffer
   //       * texCoordBuffer
   //       * tangentBuffer
   //       * bitangentBuffer
   //       * indexBuffer
   //       * hasTangentsAndBitangents
   //
   //
   //      32 RT_PROGRAM void mesh_intersect(int primIdx)
   //      33 {
   //      34     int3 index = indexBuffer[primIdx];
   //      35 
   //      36     float3 p0 = vertexBuffer[index.x];
   //      37     float3 p1 = vertexBuffer[index.y];
   //      38     float3 p2 = vertexBuffer[index.z];
   //      39 
   //      40     // Intersect ray with triangle
   //      41     float3 n;
   //      42     float  t, beta, gamma;
   //                                   _____in_______     _______out________
   //      43     if(intersect_triangle(ray, p0, p1, p2,     n, t, beta, gamma))  // from optixu_math_namespace.h
   //      44     {
   //      45         if(rtPotentialIntersection( t ))  // true means that the parametric t, may be the closest hit  
   //      46         {
   //      ..
   //      ..             .... setting attributes ...
   //      ..  
   //      88             rtReportIntersection(0);  
   //      ..                   argument specifies material index of primitive primIdx  
   //      ..                   zero when only one material for the geometry
   //      ..
   //      89         }
   //      90     }
   //
   //
   //      #. attributes calculated in mesh_intersect are available in the closest_hit and any_hit programs
   //
   //         * textureCoordinate
   //         * geometricNormal
   //         * shadingNormal
   //         * tangent
   //         * bitangent
   //
   //

    unsigned int numFaces = mesh->mNumFaces;
    unsigned int numVertices = mesh->mNumVertices;

    optix::Geometry geometry = m_context->createGeometry();

    geometry->setPrimitiveCount(numFaces);

    const char* filename = "TriangleMesh.cu" ;   // cached program is returned after creation at first call
    optix::Program intersectionProgram = m_program->createProgram( filename, "mesh_intersect" );
    optix::Program boundingBoxProgram = m_program->createProgram( filename, "mesh_bounds" );

    geometry->setIntersectionProgram(intersectionProgram);
    geometry->setBoundingBoxProgram(boundingBoxProgram);

    // Create vertex, normal and texture buffer

    optix::Buffer vertexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
    optix::float3* vertexBuffer_Host = static_cast<optix::float3*>( vertexBuffer->map() );

    optix::Buffer normalBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
    optix::float3* normalBuffer_Host = static_cast<optix::float3*>( normalBuffer->map() );

    geometry["vertexBuffer"]->setBuffer(vertexBuffer);
    geometry["normalBuffer"]->setBuffer(normalBuffer);

    // Copy vertex and normal buffers

    memcpy( static_cast<void*>( vertexBuffer_Host ),
        static_cast<void*>( mesh->mVertices ),
        sizeof( optix::float3 )*numVertices); 
    vertexBuffer->unmap();

    memcpy( static_cast<void*>( normalBuffer_Host ),
        static_cast<void*>( mesh->mNormals),
        sizeof( optix::float3 )*numVertices); 
    normalBuffer->unmap();

    // Transfer texture coordinates to buffer
    optix::Buffer texCoordBuffer;
    if(mesh->HasTextureCoords(0))
    {
        texCoordBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, numVertices);
        optix::float2* texCoordBuffer_Host = static_cast<optix::float2*>( texCoordBuffer->map());
        for(unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            aiVector3D texCoord = (mesh->mTextureCoords[0])[i];
            texCoordBuffer_Host[i].x = texCoord.x;
            texCoordBuffer_Host[i].y = texCoord.y;
        }
        texCoordBuffer->unmap();
    }
    else
    {
        texCoordBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);
    }

    geometry["texCoordBuffer"]->setBuffer(texCoordBuffer);

    // Tangents and bi-tangents buffers

    geometry["hasTangentsAndBitangents"]->setUint(mesh->HasTangentsAndBitangents() ? 1 : 0);
    if(mesh->HasTangentsAndBitangents())
    {
        optix::Buffer tangentBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
        optix::float3* tangentBuffer_Host = static_cast<optix::float3*>( tangentBuffer->map() );
        memcpy( static_cast<void*>( tangentBuffer_Host ),
            static_cast<void*>( mesh->mTangents),
            sizeof( optix::float3 )*numVertices); 
        tangentBuffer->unmap();

        optix::Buffer bitangentBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
        optix::float3* bitangentBuffer_Host = static_cast<optix::float3*>( bitangentBuffer->map() );
        memcpy( static_cast<void*>( bitangentBuffer_Host ),
            static_cast<void*>( mesh->mBitangents),
            sizeof( optix::float3 )*numVertices); 
        bitangentBuffer->unmap();

        geometry["tangentBuffer"]->setBuffer(tangentBuffer);
        geometry["bitangentBuffer"]->setBuffer(bitangentBuffer);
    }
    else
    {
        optix::Buffer emptyBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
        geometry["tangentBuffer"]->setBuffer(emptyBuffer);
        geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    }

    // Create index buffer

    optix::Buffer indexBuffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, numFaces );
    optix::int3* indexBuffer_Host = static_cast<optix::int3*>( indexBuffer->map() );
    geometry["indexBuffer"]->setBuffer(indexBuffer);

    // Copy index buffer from host to device

    for(unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        aiFace face = mesh->mFaces[i];
        indexBuffer_Host[i].x = face.mIndices[0];
        indexBuffer_Host[i].y = face.mIndices[1];
        indexBuffer_Host[i].z = face.mIndices[2];
    }

    indexBuffer->unmap();

    return geometry;
}



optix::float3  OptiXAssimpGeometry::getCenter()
{
    aiVector3D* p = AssimpGeometry::getCenter();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3  OptiXAssimpGeometry::getExtent()
{
    aiVector3D* p = AssimpGeometry::getExtent();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3 OptiXAssimpGeometry::getUp()
{
    aiVector3D* p = AssimpGeometry::getUp();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3 OptiXAssimpGeometry::getMin()
{
    aiVector3D* p = AssimpGeometry::getLow();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3 OptiXAssimpGeometry::getMax()
{
    aiVector3D* p = AssimpGeometry::getHigh();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::Aabb OptiXAssimpGeometry::getAabb()
{
    return optix::Aabb(getMin(), getMax()); 
}



