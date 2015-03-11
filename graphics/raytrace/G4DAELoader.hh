// adapted from /Developer/OptiX/SDK/sutil/PlyLoader.h  
#pragma once
#include <sutil.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <string>


class GGeo ; 

class G4DAELoader
{
public:
  G4DAELoader( const std::string&   filename,
                      optix::Context       context,
                      optix::GeometryGroup geometry_group,
                      optix::Material      material,
                      const char* ASBuilder   = "Sbvh",
                      const char* ASTraverser = "Bvh",
                      const char* ASRefine    = "0",
                      bool large_geom = false);

  ~G4DAELoader() {} // makes sure CRT objects are destroyed on the correct heap

  void setIntersectProgram( optix::Program program );

  GGeo* getGGeo(); 

  void load();
  void load( const optix::Matrix4x4& transform );

  static const char* identityFilename(char* path);  

  optix::Aabb getSceneBBox()const { return m_aabb; }

  static bool isMyFile( const std::string& filename );

private:
  void createGeometryInstance( unsigned nverts, optix::float3 const* verts, optix::float3 const* normals,
    unsigned ntris, optix::int3 const* tris );

  std::string            m_filename;

  optix::Context         m_context;
  optix::GeometryGroup   m_geometry_group;
  optix::Material        m_material;
  bool                   m_large_geom;
  const char*            m_ASBuilder;
  const char*            m_ASTraverser;
  const char*            m_ASRefine;

  optix::Aabb            m_aabb;
  GGeo*                  m_ggeo ; 

};
