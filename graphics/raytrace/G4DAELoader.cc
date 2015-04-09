// adapted from /Developer/OptiX/SDK/sutil/PlyLoader.cpp

#include "G4DAELoader.hh"

#include <optixu/optixu.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <iostream>
#include <cstring> //memcpy
#include <algorithm>

#include "AssimpWrap/AssimpCommon.hh"
#include "AssimpWrap/AssimpGeometry.hh"

#ifdef DEPRECATED
#include "AssimpOptiXGeometry.hh"
#endif

#include "AssimpWrap/AssimpGGeo.hh"
#include "GGeoOptiXGeometry.hh"

#include <stdio.h>
#include <stdlib.h>
#include "assert.h"


std::string getExtension( const std::string& filename )
{
    std::string::size_type extension_index = filename.find_last_of( "." );
    return extension_index != std::string::npos ?
           filename.substr( extension_index+1 ) : 
           std::string();
}

const char* G4DAELoader::identityFilename( char* arg )
{
     return AssimpGeometry::identityFilename(arg);
}

G4DAELoader::G4DAELoader( const std::string&   filename,
                      optix::Context       context,
                      optix::GeometryGroup geometry_group,
                      optix::Material      material,
                      const char* ASBuilder,
                      const char* ASTraverser,
                      const char* ASRefine,
                      bool large_geom )
: m_filename( filename ),
  m_context( context ),
  m_geometry_group( geometry_group ),
  m_material( material ),
  m_large_geom( large_geom),
  m_ASBuilder  (ASBuilder),
  m_ASTraverser(ASTraverser),
  m_ASRefine   (ASRefine),
  m_ggeo(NULL)
{
  // Error checking on context and geometrygroup done in ModelLoader

  assert(material.get()); // MUST DEFINE MATERIAL ELSEWHERE
}



GGeo* G4DAELoader::getGGeo()
{
   return m_ggeo ; 
}

void G4DAELoader::load()
{
  load( optix::Matrix4x4::identity() );
}


void G4DAELoader::load( const optix::Matrix4x4& transform )
{
  char* query = getenv("RAYTRACE_QUERY");     
  printf("G4DAELoader::load query %s \n", query );
  printf("G4DAELoader::load filename %s \n", m_filename.c_str() );

  // geometry selection query string is encoded into m_filename
  // the digest of the query is removed from m_filename 
  // into order to get the real path to load 
  //
  // This trickery is used in order for separate 
  // optix accelcache to be layed down for 
  // different geometry selections 

  char* path = new char[m_filename.size()]; 
  removeField( path , m_filename.c_str(), '.' , -2 );

  printf("G4DAELoader::load path     %s \n", path );

  AssimpGeometry ageo(path);
  ageo.import();
  AssimpSelection* selection = ageo.select(query);


  const char* ggctrl = getenv("RAYTRACE_GGCTRL");
  if(ggctrl) 
  {
      printf("G4DAELoader::load with GGeo intermediary using AssimpGGeo + GGeoOptixGeometry %s \n", ggctrl );

      AssimpGGeo agg(ageo.getTree(), selection); 
      m_ggeo = agg.convert(ggctrl);

      GGeoOptiXGeometry geom(m_ggeo);
      
      geom.setGeometryGroup(m_geometry_group);
      geom.setContext(m_context);   
      geom.setOverrideMaterial(m_material);  

      geom.convert(); 
      geom.setupAcceleration();

      m_aabb = geom.getAabb();
  }
  else
  {
      printf("G4DAELoader::load with AssimpOptixGeometry IS DEPRECATED\n");
#ifdef DEPRECATED
      AssimpOptiXGeometry geom(&ageo, selection);

      geom.setGeometryGroup(m_geometry_group);
      geom.setContext(m_context);  
      geom.setOverrideMaterial(m_material);  

      geom.convert(); 
      geom.setupAcceleration();

      m_aabb = geom.getAabb();
#endif
  }

  
}



void G4DAELoader::createGeometryInstance( 
  unsigned nverts, 
  optix::float3 const* verts, 
  optix::float3 const* normals, 
  unsigned ntris, optix::int3 const* tris )
{
  assert(0); // just checking if this in use

  if (m_large_geom) 
  {
    if( m_ASBuilder == std::string("Sbvh") || m_ASBuilder == std::string("KdTree")) {
      m_ASBuilder = "MedianBvh";
      m_ASTraverser = "Bvh";
    }

    RTgeometry geometry;
    unsigned int usePTX32InHost64 = 0;

    rtuCreateClusteredMesh( m_context->get(), usePTX32InHost64, &geometry, nverts, 
                           (const float*)verts, ntris, (const unsigned int*)tris, 0 );
    optix::Geometry mesh = optix::Geometry::take(geometry);
   
    optix::GeometryInstance instance = m_context->createGeometryInstance( mesh, &m_material, &m_material+1 );

    optix::Acceleration acceleration = m_context->createAcceleration(m_ASBuilder, m_ASTraverser);
    acceleration->setProperty( "refine", m_ASRefine );
    acceleration->setProperty( "leaf_size", "1" );
    acceleration->markDirty();

    m_geometry_group->setAcceleration( acceleration );
    m_geometry_group->setChildCount( 1u );
    m_geometry_group->setChild( 0, instance );

  }
  else 
  {
    optix::Buffer vertex_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, nverts );
    memcpy( vertex_buffer->map(), verts, sizeof( optix::float3 )*nverts );
    vertex_buffer->unmap();

    optix::Buffer vindex_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, ntris );
    memcpy( vindex_buffer->map(), tris, sizeof( optix::int3 )*ntris );
    vindex_buffer->unmap();

    optix::Buffer normal_buffer;
    optix::Buffer nindex_buffer;

    if( normals ) 
    {
      normal_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, nverts );
      memcpy( normal_buffer->map(), normals, sizeof( optix::float3 )*nverts );
      normal_buffer->unmap();
      nindex_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, ntris );
      memcpy( nindex_buffer->map(), tris, sizeof( optix::int3 )*ntris );
      nindex_buffer->unmap();
    }
    std::string ptx_path = std::string( sutilSamplesPtxDir() ) + (normals ? 
      "/cuda_compile_ptx_generated_triangle_mesh.cu.ptx" :
      "/cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx");

    optix::Geometry mesh = m_context->createGeometry();
    mesh->setPrimitiveCount( ntris );
    mesh->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path, "mesh_intersect" ) );
    mesh->setBoundingBoxProgram(  m_context->createProgramFromPTXFile( ptx_path, "mesh_bounds" ) );
    mesh[ "vertex_buffer" ]->set( vertex_buffer );
    mesh[ "vindex_buffer" ]->set( vindex_buffer );
    if( normals ) {
      mesh[ "normal_buffer" ]->set( normal_buffer );
      mesh[ "nindex_buffer" ]->set( nindex_buffer );
      
      // dummy buffers for unused attributes
      mesh[ "texcoord_buffer" ]->set( m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 1) );
      optix::Buffer dummy_indices = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3, ntris );
      memset( dummy_indices->map(), ~0, ntris * sizeof( optix::int3 ));
      dummy_indices->unmap();
      mesh[ "tindex_buffer" ]->set( dummy_indices );

      optix::Buffer material_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, ntris );
      memset( material_buffer->map(), 0, ntris * sizeof( int ));
      material_buffer->unmap();
      mesh[ "material_buffer" ]->set( material_buffer );
    }

    optix::GeometryInstance instance = m_context->createGeometryInstance( mesh, &m_material, &m_material+1 );

    optix::Acceleration acceleration = m_context->createAcceleration(m_ASBuilder, m_ASTraverser);
    acceleration->setProperty( "refine", m_ASRefine );
    if ( m_ASBuilder   == std::string("Sbvh")           ||
         m_ASBuilder   == std::string("TriangleKdTree") ||
         m_ASTraverser == std::string( "KdTree")        )
    {
      acceleration->setProperty( "vertex_buffer_name", "vertex_buffer" );
      acceleration->setProperty( "index_buffer_name", "vindex_buffer" );
    }
    acceleration->markDirty();

    m_geometry_group->setAcceleration( acceleration );
    m_geometry_group->setChildCount( 1u );
    m_geometry_group->setChild( 0, instance );
  }
}


bool G4DAELoader::isMyFile( const std::string& filename )
{
  return getExtension( filename ) == "dae";
}

