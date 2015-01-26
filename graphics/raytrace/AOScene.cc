#include "AOScene.hh"

#include <string.h>
#include <stdlib.h>

#include <sstream>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>


#include <optixu/optixu_vector_types.h>




AOScene::AOScene(const char* path, const char* ptxfold, const char* target)
        : 
        SampleScene(),
        m_scene(NULL),
        m_path(NULL),
        m_ptxfold(NULL),
        m_target(NULL),
        m_width(1080u),
        m_height(720u),
        m_importer(new Assimp::Importer()),
        m_intersectionProgram(NULL),
        m_boundingBoxProgram(NULL)
{
    if(!path || !ptxfold || !target) return ; 
    printf("AOScene::AOScene ctor path %s ptxfold %s target %s \n", path, ptxfold, target);
    m_path = strdup(path);
    m_ptxfold = strdup(ptxfold);
    m_target = strdup(target);
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


const char* const AOScene::ptxpath( const std::string& base )
{
  static std::string path;
  path = generated_ptx_path(m_ptxfold, m_target, base.c_str());
  return path.c_str();
}


optix::Program AOScene::createProgram(const char* filename, const char* fname )
{
  std::string path = ptxpath(filename); 
  std::string key = path + ":" + fname ; 

  if(m_programs.find(key) == m_programs.end())
  { 
       printf("createProgram key %s \n", key.c_str() );
       optix::Program program = m_context->createProgramFromPTXFile( path.c_str(), fname ); 
       m_programs[key] = program ; 
  } 
  else
  {
       //printf("createProgram cached key %s \n", key.c_str() );
  } 
  return m_programs[key];
}


void AOScene::initScene( InitialCameraData& camera_data )
{
  // set up path to ptx file associated with tutorial number
  std::stringstream ss;
  ss << "tutorial0.cu";
  m_ptx_path = ptxpath( ss.str() );

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
        optix::Material material = convertMaterial(m_scene->mMaterials[i]);
        m_materials.push_back(material);
    }

    for(unsigned int i = 0; i < m_scene->mNumMeshes; i++)
    {
        optix::Geometry geometry = convertGeometry(m_scene->mMeshes[i]);
        m_geometries.push_back(geometry);
    }


    optix::Group root = convertNode(m_scene->mRootNode);

    // double egging pudding  ?
    optix::Acceleration acceleration = m_context->createAcceleration("Sbvh", "Bvh"); 
    root->setAcceleration( acceleration );
    acceleration->markDirty();

    m_context["top_object"]->set(root);
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



void AOScene::dumpMaterial(aiMaterial* ai_material)
{
    aiString name;
    ai_material->Get(AI_MATKEY_NAME, name);
    unsigned int numProperties = ai_material->mNumProperties ;
    printf("AOScene::dumpMaterial props %2d %s \n", numProperties, name.C_Str());

    for(unsigned int i = 0; i < ai_material->mNumProperties; i++)
    {
        aiMaterialProperty* property = ai_material->mProperties[i] ;
        aiString key = property->mKey ; 
        printf("key %s \n", key.C_Str());
    }
}

optix::Group AOScene::convertNode(aiNode* node)
{
    if(node->mNumMeshes > 0)
    {  
        // all node meshes into single geometry group of geometry instances
        optix::GeometryGroup gg = m_context->createGeometryGroup();
        gg->setChildCount(node->mNumMeshes);
        for(unsigned int i = 0; i < node->mNumMeshes; i++)
        {   
            unsigned int meshIndex = node->mMeshes[i];
            aiMesh* mesh = m_scene->mMeshes[meshIndex];
            unsigned int materialIndex = mesh->mMaterialIndex;

            optix::Geometry geometry = m_geometries[meshIndex] ;
            optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, m_materials.begin()+materialIndex , m_materials.begin() + materialIndex + 1  );

            gg->setChild(i, gi);
        }   

        {   
            optix::Acceleration acceleration = m_context->createAcceleration("Sbvh", "Bvh");
            acceleration->setProperty( "vertex_buffer_name", "vertexBuffer" );
            acceleration->setProperty( "index_buffer_name", "indexBuffer" );
            gg->setAcceleration( acceleration );
            acceleration->markDirty();
        }   

        optix::Group group = m_context->createGroup();
        group->setChildCount(1);
        group->setChild(0, gg);
        {   
            optix::Acceleration acceleration = m_context->createAcceleration("NoAccel", "NoAccel");
            group->setAcceleration( acceleration );
        }   
        return group;
    }   
    else if(node->mNumChildren > 0)
    {
        std::vector<optix::Group> groups;
        for(unsigned int i = 0; i < node->mNumChildren; i++)
        {
            aiNode* childNode = node->mChildren[i];
            optix::Group childGroup = convertNode(childNode);
            if(childGroup)
            {
                groups.push_back(childGroup);
            }
        }

        if(groups.size() > 0)
        {
            optix::Group group = m_context->createGroup(groups.begin(), groups.end());
            optix::Acceleration acceleration = m_context->createAcceleration("Sbvh", "Bvh");
            group->setAcceleration( acceleration );
            return group;
        }
    }

    optix::Group empty = m_context->createGroup();
    optix::Acceleration acceleration = m_context->createAcceleration("NoAccel", "NoAccel");
    empty->setAcceleration(acceleration);
    return empty;
}


optix::Material AOScene::convertMaterial(aiMaterial* ai_material)
{
    /*
    TODO:
        get assimp to access wavelength dependant material properties
        and feed them through into the material program 
        * by code gen of tables ? referencing a buffer of structs ?
    */

    optix::Material material = m_context->createMaterial();
    const char* filename = "material0.cu" ; 
    const char* fname = "closest_hit_radiance" ; 
    optix::Program  program = createProgram(filename, fname);
    material->setClosestHitProgram(0, program);
    material["Kd"]->setFloat( 0.7f, 0.7f, 0.7f);
    return material ; 
}


optix::Geometry AOScene::convertGeometry(aiMesh* mesh)
{
    unsigned int numFaces = mesh->mNumFaces;
    unsigned int numVertices = mesh->mNumVertices;

    optix::Geometry geometry = m_context->createGeometry();

    geometry->setPrimitiveCount(numFaces);
    geometry->setIntersectionProgram(m_intersectionProgram);
    geometry->setBoundingBoxProgram(m_boundingBoxProgram);

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


