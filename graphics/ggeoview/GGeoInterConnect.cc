#include "GGeoInterConnect.hh"

#include "assert.h"
#include "stdio.h"

#include "Buffer.hh"
#include "GMergedMesh.hh"
#include "GVector.hh"

GGeoInterConnect::GGeoInterConnect(GMergedMesh* mm) :
            m_num_elements(mm->getNumFaces()),
            m_vertices(NULL),
            m_colors(NULL),
            m_indices(NULL),
            IGeometry()
{
    assert(sizeof(gfloat3) == sizeof(float)*3);
    assert(sizeof(guint3) == sizeof(unsigned int)*3);

    m_vertices = new Buffer( sizeof(gfloat3)*mm->getNumVertices(), (void*)mm->getVertices()) ;
    m_colors   = new Buffer( sizeof(gfloat3)*mm->getNumColors(),   (void*)mm->getColors()) ;
    m_indices  = new Buffer( sizeof(guint3)*mm->getNumFaces(),     (void*)mm->getFaces()) ;
}

GGeoInterConnect::~GGeoInterConnect()
{}

unsigned int GGeoInterConnect::getNumElements()
{
    return m_num_elements ;
}
Buffer* GGeoInterConnect::getVertices()
{
    return m_vertices ;
}
Buffer* GGeoInterConnect::getColors()
{
    return m_colors ;
}
Buffer* GGeoInterConnect::getIndices()
{
    return m_indices ;
}


void GGeoInterConnect::Summary(const char* msg)
{
    printf("%s NumElements %u \n", msg, getNumElements() );

    m_vertices->Summary("GGeoInterConnect::Summary vertices ");
    m_colors->Summary("GGeoInterConnect::Summary colors ");
    m_indices->Summary("GGeoInterConnect::Summary indices ");
}




