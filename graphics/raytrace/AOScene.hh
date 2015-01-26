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

#include <string>
#include <map>


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
    AOScene(const char* path, const char* ptxfold, const char* target, const char* query );
    virtual ~AOScene();
    const char* const ptxpath( const std::string& filename );

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
    void Info();

    aiNode* searchNode(const char* query);

private:
    void dumpMaterial(aiMaterial* ai_material);


    optix::Material convertMaterial(aiMaterial* ai_material);

    optix::Geometry convertGeometry(aiMesh* mesh);

    optix::Group convertNode(aiNode* node);

private:
    char* m_path ; 

    char* m_ptxfold ; 

    char* m_target ; 

    char* m_query ; 

    unsigned int m_width  ;

    unsigned int m_height ;

    std::string  m_ptx_path;

private:

    Assimp::Importer* m_importer;

    const aiScene* m_scene;

    optix::Program m_intersectionProgram;

    optix::Program m_boundingBoxProgram;

private:

    std::vector<optix::Material> m_materials;

    std::vector<optix::Geometry> m_geometries;

    std::map<std::string,optix::Program> m_programs;

};


#endif

