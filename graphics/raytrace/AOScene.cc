#include "AOScene.hh"

#include <string.h>
#include <stdlib.h>

#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>


#include <optixu/optixu_vector_types.h>


const char* AOScene::TARGET = "RayTrace" ; 


AOScene::AOScene(const char* path)
        : 
        SampleScene(),
        m_scene(NULL),
        m_path(NULL),
        m_width(1080u),
        m_height(720u),
        m_importer(new Assimp::Importer()),
        m_intersectionProgram(NULL),
        m_boundingBoxProgram(NULL)
{
    if(!path) return ; 
    printf("AOScene ctor path %s \n", path);
    m_path = strdup(path);
}

AOScene::~AOScene(void)
{
    printf("AOScene dtor\n");

    // deleting m_importer also deletes the scene
    delete m_importer;

    free(m_path);
}


std::string generated_ptx_path( const char* folder, const char* target, const char* base )
{
  std::stringstream ss;
  ss << folder << "/" << target << "_generated_" << base << ".ptx" ;
  return ss.str() ;
}


const char* const AOScene::ptxpath( const std::string& target, const std::string& base )
{
  // TODO: anchor this to avoid invoking dir dependency 
  static std::string path;
  path = generated_ptx_path(".", target.c_str(), base.c_str());
  return path.c_str();
}


optix::Program AOScene::createProgram(const char* filename, const char* fname )
{
  std::string path = generated_ptx_path(".", TARGET, filename ); 
  printf("createProgram target %s filename %s fname %s => path %s \n", TARGET, filename, fname, path.c_str() );
  return m_context->createProgramFromPTXFile( path.c_str(), fname ); 
}



void AOScene::initScene( InitialCameraData& camera_data )
{
  // set up path to ptx file associated with tutorial number
  std::stringstream ss;
  ss << "tutorial0.cu";
  m_ptx_path = ptxpath( "RayTrace", ss.str() );

  printf("AOScene::initScene m_ptx_path %s \n", m_ptx_path.c_str());

  m_context->setRayTypeCount( 1 );
  m_context->setEntryPointCount( 1 );
  m_context->setStackSize( 4640 );
  
  m_context["max_depth"]->setInt(100);
  m_context["radiance_ray_type"]->setUint(0);
  m_context["shadow_ray_type"]->setUint(1);
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-3f );
  m_context["importance_cutoff"]->setFloat( 0.01f );
  m_context["ambient_light_color"]->setFloat( 0.31f, 0.33f, 0.28f );

  m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height) );


  const char* filename = "tutorial0.cu" ; 
  optix::Program ray_gen_program = createProgram(filename, "pinhole_camera" );  
  optix::Program exception_program = createProgram(filename, "exception" );
  optix::Program miss_program = createProgram(filename, "miss" );
 
  m_context->setRayGenerationProgram( 0, ray_gen_program ); 
  m_context->setExceptionProgram( 0, exception_program );
  m_context->setMissProgram( 0, miss_program );

  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context["bg_color"]->setFloat( optix::make_float3( 0.34f, 0.55f, 0.85f ) );
  

 // Set up camera
  camera_data = InitialCameraData( optix::make_float3( 7.0f, 9.2f, -6.0f ), // eye
                                   optix::make_float3( 0.0f, 4.0f,  0.0f ), // lookat
                                   optix::make_float3( 0.0f, 1.0f,  0.0f ), // up
                                   60.0f );                          // vfov

  m_context["eye"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( optix::make_float3( 0.0f, 0.0f, 0.0f ) );


  initGeometry(m_context);


  printf("context validate\n");
  m_context->validate();
  printf("context compile\n");
  m_context->compile();
  printf("context compile DONE\n");

}


optix::Buffer AOScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}

void AOScene::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
}




void AOScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  optix::Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, static_cast<unsigned int>(buffer_width),
                      static_cast<unsigned int>(buffer_height) );
}


void AOScene::initGeometry(optix::Context& context)
{
    if(!m_intersectionProgram)
    {   
        const char* filename = "TriangleMesh.cu" ; 
        m_intersectionProgram = createProgram( filename, "mesh_intersect" );
        m_boundingBoxProgram = createProgram( filename, "mesh_bounds" );
    }    

    m_scene = m_importer->ReadFile( m_path, 
                     aiProcess_CalcTangentSpace       |   
                     aiProcess_Triangulate            |   
                     aiProcess_JoinIdenticalVertices  |
                     aiProcess_SortByPType);

    if(!m_scene)
    {   
        printf("import error : %s \n", m_importer->GetErrorString() );  
    }   


    for(unsigned int i = 0; i < m_scene->mNumMaterials; i++)
    {
        aiMaterial* material = m_scene->mMaterials[i];
        LoadMaterial(material);
    }



    std::vector<optix::Geometry> geometries;
    for(unsigned int i = 0; i < m_scene->mNumMeshes; i++)
    {
        aiMesh* mesh = m_scene->mMeshes[i] ;
        optix::Geometry geometry = createGeometryFromMesh(mesh, context);
        geometries.push_back(geometry);
    }

}

void AOScene::Info()
{
    printf("scene %p \n", m_scene);
    if(!m_scene) return ; 
    printf("scene Flags         %d \n", m_scene->mFlags );
    printf("scene NumAnimations %d \n", m_scene->mNumAnimations );
    printf("scene NumCameras    %d \n", m_scene->mNumCameras );
    printf("scene NumLights     %d \n", m_scene->mNumLights );
    printf("scene NumMaterials  %d \n", m_scene->mNumMaterials );
    printf("scene NumMeshes     %d \n", m_scene->mNumMeshes );
    printf("scene NumTextures   %d \n", m_scene->mNumTextures );
}


void AOScene::LoadMaterial(aiMaterial* material)
{
    aiString name;
    material->Get(AI_MATKEY_NAME, name);

    unsigned int numProperties = material->mNumProperties ;

    printf("Material props %2d %s \n", numProperties, name.C_Str());

    for(unsigned int i = 0; i < material->mNumProperties; i++)
    {
        aiMaterialProperty* property = material->mProperties[i] ;

        //aiString key = property->mKey ; 
        //printf("key %s \n", key.C_Str());
    }
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


