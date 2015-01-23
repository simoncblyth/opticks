/* 
   Intersection of Assimp and OptiX

   Some aspects inspired by 
       /usr/local/env/cuda/optix/OppositeRenderer/OppositeRenderer/RenderEngine/scene/Scene.h 
*/

#ifndef AOSCENE_H
#define AOSCENE_H

#include <optixu/optixpp_namespace.h>

struct aiScene;
struct aiMesh;
struct aiNode;

namespace Assimp
{
    class Importer;
}

class AOScene 
{
public:
    AOScene(const char* path);

    virtual ~AOScene();

private:
    optix::Geometry createGeometryFromMesh(aiMesh* mesh, optix::Context& context);


private:
    char* m_path ; 

    Assimp::Importer* m_importer;

    const aiScene* m_scene;

    optix::Program m_intersectionProgram;

    optix::Program m_boundingBoxProgram;

};


#endif

