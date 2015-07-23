#include "OptiXGeometry.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>

#include <optixu/optixu_vector_types.h>


OptiXGeometry::~OptiXGeometry()
{
}

OptiXGeometry::OptiXGeometry()
           : 
           m_context(NULL),
           m_override_material(NULL),
           m_geometry_group(NULL)
{
}

void OptiXGeometry::setContext(optix::Context& context)
{
    m_context = context ;   
}
void OptiXGeometry::setOverrideMaterial(optix::Material material)
{
    m_override_material = material ;   
}
void OptiXGeometry::setGeometryGroup(optix::GeometryGroup gg)
{
    m_geometry_group = gg ; 
}


optix::Context OptiXGeometry::getContext()
{
    return m_context ;
}
optix::Material OptiXGeometry::getOverrideMaterial()
{
    return m_override_material ; 
}
optix::GeometryGroup OptiXGeometry::getGeometryGroup()
{
    return m_geometry_group ; 
}


optix::Material OptiXGeometry::getMaterial(unsigned int index)
{
    printf("OptiXGeometry::getMaterial index %u size %lu \n", index, m_materials.size() );  
    assert(index < m_materials.size());
    return m_materials[index] ; 
}


void OptiXGeometry::setupAcceleration()
{
    //const char* builder = "Sbvh" ;
    const char* builder = "Bvh" ;
    const char* traverser = "Bvh" ;

    LOG(info) << "OptiXGeometry::setupAcceleration for " 
              << " gis " <<  m_gis.size() 
              << " builder " << builder 
              << " traverser " << traverser
              ; 
    
    optix::Acceleration acceleration = m_context->createAcceleration(builder, traverser);
    acceleration->setProperty( "vertex_buffer_name", "vertexBuffer" );
    acceleration->setProperty( "index_buffer_name", "indexBuffer" );

    m_geometry_group->setAcceleration( acceleration );

    acceleration->markDirty();

    m_geometry_group->setChildCount(m_gis.size());
    for(unsigned int i=0 ; i <m_gis.size() ; i++) m_geometry_group->setChild(i, m_gis[i]);

    // FOR UNKNOWN REASONS SETTING top_object CAUSES SEGFAULT WHEN USED FROM SEPARATE SO 
    // AND NOT WHEN ALL COMPILED INTO SAME EXECUTABLE
    // ... IT DUPLICATES A SETTING IN MeshViewer ANYHOW SO NO PROBLEM SKIPPING IT 
    //
    //  assuming a not-updated lib is the cause
    //
    //m_context["top_object"]->set(m_geometry_group);

    LOG(info) << "OptiXGeometry::setupAcceleration DONE ";
}


optix::Aabb OptiXGeometry::getAabb()
{
    return optix::Aabb(getMin(), getMax()); 
}


