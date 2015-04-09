#include "OptiXGeometry.hh"


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
    assert(index < m_materials.size());
    return m_materials[index] ; 
}


void OptiXGeometry::setupAcceleration()
{
    optix::Acceleration acceleration = m_context->createAcceleration("Sbvh", "Bvh");
    acceleration->setProperty( "vertex_buffer_name", "vertexBuffer" );
    acceleration->setProperty( "index_buffer_name", "indexBuffer" );

    m_geometry_group->setAcceleration( acceleration );

    acceleration->markDirty();

    m_geometry_group->setChildCount(m_gis.size());
    for(unsigned int i=0 ; i <m_gis.size() ; i++) m_geometry_group->setChild(i, m_gis[i]);

    m_context["top_object"]->set(m_geometry_group);
}


optix::Aabb OptiXGeometry::getAabb()
{
    return optix::Aabb(getMin(), getMax()); 
}


