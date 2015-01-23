#include "AOScene.hh"

#include <string.h>
#include <stdlib.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>

AOScene::AOScene(const char* path)
        : 
        m_scene(NULL),
        m_path(strdup(path)),
        m_importer(new Assimp::Importer()),
        m_intersectionProgram(NULL),
        m_boundingBoxProgram(NULL)
{
}

AOScene::~AOScene(void)
{
    printf("Delete scene\n");

    // deleting m_importer also deletes the scene
    delete m_importer;

    free(m_path);
}

optix::Geometry AOScene::createGeometryFromMesh(aiMesh* mesh, optix::Context& context)
{
    unsigned int numFaces = mesh->mNumFaces;
    unsigned int numVertices = mesh->mNumVertices;

    optix::Geometry geometry = context->createGeometry();

    geometry->setPrimitiveCount(numFaces);
    geometry->setIntersectionProgram(m_intersectionProgram);
    geometry->setBoundingBoxProgram(m_boundingBoxProgram);

    // Create vertex, normal and texture buffer

    optix::Buffer vertexBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
    optix::float3* vertexBuffer_Host = static_cast<optix::float3*>( vertexBuffer->map() );

    optix::Buffer normalBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
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
        texCoordBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, numVertices);
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
        texCoordBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);
    }

    geometry["texCoordBuffer"]->setBuffer(texCoordBuffer);

    // Tangents and bi-tangents buffers

    geometry["hasTangentsAndBitangents"]->setUint(mesh->HasTangentsAndBitangents() ? 1 : 0);
    if(mesh->HasTangentsAndBitangents())
    {
        optix::Buffer tangentBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
        optix::float3* tangentBuffer_Host = static_cast<optix::float3*>( tangentBuffer->map() );
        memcpy( static_cast<void*>( tangentBuffer_Host ),
            static_cast<void*>( mesh->mTangents),
            sizeof( optix::float3 )*numVertices); 
        tangentBuffer->unmap();

        optix::Buffer bitangentBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
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
        optix::Buffer emptyBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, 0);
        geometry["tangentBuffer"]->setBuffer(emptyBuffer);
        geometry["bitangentBuffer"]->setBuffer(emptyBuffer);
    }

    // Create index buffer

    optix::Buffer indexBuffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, numFaces );
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


