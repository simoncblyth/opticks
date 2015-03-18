#include "AssimpOptiXGeometry.hh"
#include "AssimpWrap/AssimpNode.hh"
#include "AssimpWrap/AssimpGeometry.hh"
#include "AssimpWrap/AssimpSelection.hh"

#include "RayTraceConfig.hh"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>

#include <optixu/optixu_vector_types.h>

#include "OptiXGeometry.hh"

AssimpOptiXGeometry::~AssimpOptiXGeometry()
{
}

AssimpOptiXGeometry::AssimpOptiXGeometry(AssimpGeometry* ageo, AssimpSelection* selection)
           : 
           OptiXGeometry(),
           m_ageo(ageo),
           m_selection(selection)
{
}


void AssimpOptiXGeometry::convert()
{
    convertMaterials();
    convertStructure();
}

void AssimpOptiXGeometry::convertMaterials()
{
    // these materials are currently ignored, the single material coming from m_material
    for(unsigned int i = 0; i < m_ageo->getNumMaterials(); i++)
    {
        optix::Material material = convertMaterial(m_ageo->getMaterial(i));
        m_materials.push_back(material);
    }
}

void AssimpOptiXGeometry::convertStructure()
{
    //  Following fig2 of documentation:  single gg containing many gi 
    //  converting aiMesh into optix::GeometryInstance collected into m_gis 


    assert(m_selection->getNumSelected() > 0);  // must select some geometry before convert
    m_gis.clear();

    bool mergemesh = m_selection->getQueryMerge();
    bool recurse = !m_selection->isFlatSelection() ; 

    if(mergemesh)
    {
        printf("AssimpOptiXGeometry::convertStructure createMergedMesh \n");

        aiMesh* mesh = m_ageo->createMergedMesh(m_selection); 

        optix::Geometry geometry = convertGeometry(mesh) ;  

        optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, &m_override_material, &m_override_material + 1  );  
     
        m_gis.push_back(gi);   

    }
    else
    {
        //  Recursive traverse down from each selected AssimpNode
        for(unsigned int i=0 ; i < m_selection->getNumSelected() ; i++ )
        {
            traverseNode(m_selection->getSelectedNode(i), 0, recurse);
        }
    } 

    printf("AssimpOptiXGeometry::convertStructure : %d selected top nodes with %lu gi \n", m_selection->getNumSelected(), m_gis.size() );
    assert(m_gis.size() > 0);
}


void AssimpOptiXGeometry::traverseNode(AssimpNode* node, unsigned int depth, bool recurse)
{
   //
   //  Recursive traverse of the AssimpNode converting aiMesh into optix::GeometryInstance 
   //  and collecting into m_gis 
   //
    int maxdepth = m_selection->getQueryDepth();

    if(depth < maxdepth )
    {
        for(unsigned int i = 0; i < node->getNumMeshes(); i++)
        {   
            aiMesh* mesh = node->getMesh(i);   // these are copied and globally positioned meshes 

            optix::Geometry geometry = convertGeometry(mesh) ;  

            optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, &m_override_material, &m_override_material + 1  );  

            m_gis.push_back(gi);   
        }   
    }

    if(recurse)
    {
        for(unsigned int i = 0; i < node->getNumChildren(); i++) traverseNode(node->getChild(i), depth + 1, recurse);
    }
}


optix::Material AssimpOptiXGeometry::convertMaterial(aiMaterial* ai_material)
{
    // NB material properties currently ignored

    RayTraceConfig* cfg = RayTraceConfig::getInstance();

    optix::Material material = m_context->createMaterial();

    material->setClosestHitProgram(0, cfg->createProgram("material1_radiance.cu", "closest_hit_radiance"));

    return material ; 
}


optix::Geometry AssimpOptiXGeometry::convertGeometry(aiMesh* mesh)
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

    RayTraceConfig* cfg = RayTraceConfig::getInstance();

    unsigned int numFaces = mesh->mNumFaces;
    unsigned int numVertices = mesh->mNumVertices;

    optix::Geometry geometry = m_context->createGeometry();
    geometry->setPrimitiveCount(numFaces);

    geometry->setIntersectionProgram(cfg->createProgram("TriangleMesh.cu", "mesh_intersect"));
    geometry->setBoundingBoxProgram(cfg->createProgram("TriangleMesh.cu", "mesh_bounds"));

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






optix::float3  AssimpOptiXGeometry::getCenter()
{
    aiVector3D* p = m_selection->getCenter();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3  AssimpOptiXGeometry::getExtent()
{
    aiVector3D* p = m_selection->getExtent();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3 AssimpOptiXGeometry::getUp()
{
    aiVector3D* p = m_selection->getUp();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3 AssimpOptiXGeometry::getMin()
{
    aiVector3D* p = m_selection->getLow();
    return optix::make_float3(p->x, p->y, p->z); 
}

optix::float3 AssimpOptiXGeometry::getMax()
{
    aiVector3D* p = m_selection->getHigh();
    return optix::make_float3(p->x, p->y, p->z); 
}



