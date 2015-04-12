#include "Geometry.hh"

// assimpwrap
#include "AssimpWrap/AssimpGGeo.hh"
#include "GMergedMesh.hh"
#include "GGeo.hh"

Geometry::Geometry() 
   :
   m_ggeo(NULL),
   m_geo(NULL)
{
}

GGeo* Geometry::getGGeo()
{
    return m_ggeo ; 
}

GMergedMesh* Geometry::getGeo()
{
    return m_geo ; 
}

GDrawable* Geometry::getDrawable()
{
    return m_geo ; 
}


void Geometry::load(const char* envprefix)
{
    m_ggeo = AssimpGGeo::load(envprefix);
    m_geo = m_ggeo->getMergedMesh(); 

    //GMesh* geo = ggeo->getMesh(0); 
    //Demo* geo = new Demo()

    assert(m_geo);
    m_geo->setColor(0.5,0.5,1.0);
}

void Geometry::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_geo->Summary("Geometry::Summary");
    m_geo->Dump("Geometry::Summary Dump",10);
}



