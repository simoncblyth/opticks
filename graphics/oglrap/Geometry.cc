#include "Geometry.hh"

// assimpwrap
#include "AssimpWrap/AssimpGGeo.hh"
#include "GMergedMesh.hh"
#include "GGeo.hh"

Geometry::Geometry() 
   :
   m_ggeo(NULL),
   m_mergedmesh(NULL)
{
}

GGeo* Geometry::getGGeo()
{
    return m_ggeo ; 
}

GMergedMesh* Geometry::getMergedMesh()
{
    return m_mergedmesh ; 
}
GDrawable* Geometry::getDrawable()
{
    return m_mergedmesh ; 
}


void Geometry::load(const char* envprefix)
{
    m_ggeo = AssimpGGeo::load(envprefix);
    m_mergedmesh = m_ggeo->getMergedMesh(); 
    assert(m_mergedmesh);
    m_mergedmesh->setColor(0.5,0.5,1.0);
}

void Geometry::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_mergedmesh->Summary("Geometry::Summary");
    m_mergedmesh->Dump("Geometry::Summary Dump",10);
}



