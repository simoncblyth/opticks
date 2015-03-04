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
           m_material(NULL),
           m_geometry_group(NULL)
{
}

void OptiXGeometry::setContext(optix::Context& context)
{
    m_context = context ;   
}
void OptiXGeometry::setMaterial(optix::Material material)
{
    m_material = material ;   
}
void OptiXGeometry::setGeometryGroup(optix::GeometryGroup gg)
{
    m_geometry_group = gg ; 
}


optix::Context OptiXGeometry::getContext()
{
    return m_context ;
}
optix::Material OptiXGeometry::getMaterial()
{
    return m_material ; 
}
optix::GeometryGroup OptiXGeometry::getGeometryGroup()
{
    return m_geometry_group ; 
}



void OptiXGeometry::addInstance(optix::Geometry geometry, optix::Material material)
{
     optix::GeometryInstance gi = m_context->createGeometryInstance( geometry, &material, &material+1  );   // single material 

     m_gis.push_back(gi);
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


