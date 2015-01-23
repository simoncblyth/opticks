/* 
   Intersection of Assimp and OptiX

   Some aspects inspired by 
       /usr/local/env/cuda/optix/OppositeRenderer/OppositeRenderer/RenderEngine/scene/Scene.h 
*/

#ifndef AOSCENE_H
#define AOSCENE_H

#include <GLUTDisplay.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>


struct aiScene;
struct aiMesh;
struct aiNode;
struct aiMaterial; 

namespace Assimp
{
    class Importer;
}


// following Tutorial from optix-tutorial-vi
class AOScene  : public SampleScene 
{
public:
    AOScene(const char* path);
    virtual ~AOScene();
    static const char* const ptxpath( const std::string& target, const std::string& base );

    static const char* TARGET ; 

public:
   // From SampleScene
   void   initScene( InitialCameraData& camera_data );
   void   trace( const RayGenCameraData& camera_data );
   void   doResize( unsigned int width, unsigned int height );

   void   setDimensions( const unsigned int w, const unsigned int h ) { m_width = w; m_height = h; }
   optix::Buffer getOutputBuffer();

public:
    optix::Program createProgram(const char* filename, const char* fname );
    void initGeometry(optix::Context& context);
    void LoadMaterial(aiMaterial* material);
    void Info();

private:
    optix::Geometry createGeometryFromMesh(aiMesh* mesh, optix::Context& context);

private:
    char* m_path ; 

    unsigned int m_width  ;

    unsigned int m_height ;

    std::string  m_ptx_path;

private:

    Assimp::Importer* m_importer;

    const aiScene* m_scene;

    optix::Program m_intersectionProgram;

    optix::Program m_boundingBoxProgram;

};


#endif

