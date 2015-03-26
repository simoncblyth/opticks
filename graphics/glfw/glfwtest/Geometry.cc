#include "Geometry.hh"
#include "stdio.h"

const float Geometry::pvertex[] = {
   0.0f,  0.5f,  0.0f,
   0.5f, -0.5f,  0.0f,
  -0.5f, -0.5f,  0.0f
};

const float Geometry::pcolor[] = {
  1.0f, 0.0f,  0.0f,
  0.0f, 1.0f,  0.0f,
  0.0f, 0.0f,  1.0f
};

const unsigned int Geometry::pindex[] = {
      0,  1,  2
};

void Geometry::load_defaults()
{
    m_vertices = new Array<float>(9, &pvertex[0]);
    m_colors   = new Array<float>(9, &pcolor[0]);
    m_indices  = new Array<unsigned int>(3,  &pindex[0]);
}

Geometry::Geometry() :
    IGeometry(),
    m_vertices(NULL),
    m_colors(NULL),
    m_indices(NULL)
{
}

Geometry::~Geometry()
{
}

unsigned int Geometry::getNumElements()
{
    return m_indices ? m_indices->getLength() : 0 ; 
}

Array<float>* Geometry::getVertices()
{
    return m_vertices ; 
}
Array<float>* Geometry::getColors()
{
    return m_colors ; 
}
Array<unsigned int>* Geometry::getIndices()
{
    return m_indices ; 
}

void Geometry::load(const char* path)
{
    if(!path)
    {
        load_defaults();
    }
    else
    {
         printf("Geometry::load %s \n", path );
    }
}



