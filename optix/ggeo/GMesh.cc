#include "GMesh.hh"
#include "GBuffer.hh"
#include "stdio.h"
#include "assert.h"
#include <algorithm>

GMesh::GMesh(GMesh* other) 
     :
     m_index(other->getIndex()),
     m_vertices(other->getVertices()),
     m_vertices_buffer(other->getVerticesBuffer()),
     m_num_vertices(other->getNumVertices()),
     m_faces(other->getFaces()),
     m_faces_buffer(other->getFacesBuffer()),
     m_num_faces(other->getNumFaces()),
     m_colors(other->getColors()),
     m_colors_buffer(other->getColorsBuffer()),
     m_num_colors(other->getNumColors()),
     GDrawable()
{
   updateBounds();
}

GMesh::GMesh(unsigned int index, gfloat3* vertices, unsigned int num_vertices, guint3* faces, unsigned int num_faces) 
      :
      m_index(index),
      m_vertices(NULL),
      m_vertices_buffer(NULL),
      m_num_vertices(num_vertices), 
      m_faces(NULL),
      m_faces_buffer(NULL),
      m_num_faces(num_faces),
      m_colors(NULL),
      m_colors_buffer(NULL),
      m_low(NULL),
      m_high(NULL),
      m_dimensions(NULL),
      m_center(NULL),
      m_model_to_world(NULL),
      m_extent(0.f),
      m_num_colors(num_vertices),
      GDrawable()
{
   // not yet taking ownership, depends on continued existance of data source 

   setVertices(vertices);
   setFaces(faces);
   updateBounds();
}


unsigned int GMesh::getIndex()
{
    return m_index ; 
}
unsigned int GMesh::getNumVertices()
{
    return m_num_vertices ; 
}
unsigned int GMesh::getNumColors()
{
    return m_num_colors ;   
}
unsigned int GMesh::getNumFaces()
{
    return m_num_faces ; 
}


void GMesh::setNumColors(unsigned int num_colors)
{
   m_num_colors = num_colors ;
}






gfloat3* GMesh::getLow()
{
    return m_low ;
}
gfloat3* GMesh::getHigh()
{
    return m_high ;
}
gfloat3* GMesh::getDimensions()
{
    return m_dimensions ; 
}

GMatrix<float>* GMesh::getModelToWorld()
{
    return m_model_to_world ; 
}





gfloat3* GMesh::getVertices()
{
    return m_vertices ;
}
gfloat3* GMesh::getColors()
{
    return m_colors ;
}
guint3*  GMesh::getFaces()
{
    return m_faces ;
}


GBuffer* GMesh::getVerticesBuffer()
{
    return m_vertices_buffer ;
}
GBuffer* GMesh::getColorsBuffer()
{
    return m_colors_buffer ;
}
GBuffer*  GMesh::getFacesBuffer()
{
    return m_faces_buffer ;
}
GBuffer*  GMesh::getModelToWorldBuffer()
{
    return (GBuffer*)m_model_to_world ;
}




void GMesh::setVertices(gfloat3* vertices)
{
    m_vertices = vertices ;
    m_vertices_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_vertices ) ;
    assert(sizeof(gfloat3) == sizeof(float)*3);
}
void GMesh::setFaces(guint3* faces)
{
    m_faces = faces ;
    m_faces_buffer = new GBuffer( sizeof(guint3)*m_num_faces, (void*)m_faces ) ;
    assert(sizeof(guint3) == sizeof(unsigned int)*3);
}
void GMesh::setColors(gfloat3* colors)
{
    m_colors = colors ;
    m_colors_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_colors ) ;
}



void GMesh::setLow(gfloat3* low)
{
    m_low = low ;
}
void GMesh::setHigh(gfloat3* high)
{
    m_high = high ;
}





GMesh::~GMesh()
{
}

void GMesh::Summary(const char* msg)
{
   printf("%s idx %u vx %u fc %u \n",
      msg, 
      m_index, 
      m_num_vertices, 
      m_num_faces);

   printf("low %10.3f %10.3f %10.3f\n",
         m_low->x,
         m_low->y,
         m_low->z);

   printf("high %10.3f %10.3f %10.3f\n", 
          m_high->x,
          m_high->y,
          m_high->z);

}


void GMesh::updateBounds()
{
    gfloat3  low( 1e10f, 1e10f, 1e10f);
    gfloat3 high( -1e10f, -1e10f, -1e10f);

    for( unsigned int i = 0; i < m_num_vertices ;++i )
    {
        gfloat3& v = m_vertices[i];

        low.x = std::min( low.x, v.x);
        low.y = std::min( low.y, v.y);
        low.z = std::min( low.z, v.z);

        high.x = std::max( high.x, v.x);
        high.y = std::max( high.y, v.y);
        high.z = std::max( high.z, v.z);
    }

    m_low = new gfloat3(low.x, low.y, low.z) ;
    m_high = new gfloat3(high.x, high.y, high.z);

    m_dimensions = new gfloat3(high.x - low.x, high.y - low.y, high.z - low.z );
    m_center     = new gfloat3((high.x + low.x)/2.0f, (high.y + low.y)/2.0f , (high.z + low.z)/2.0f );
    m_extent = 0.f ;
    m_extent = std::max( m_dimensions->x , m_extent );
    m_extent = std::max( m_dimensions->y , m_extent );
    m_extent = std::max( m_dimensions->z , m_extent );
    m_extent = m_extent / 2.0f ; 

    m_model_to_world = new GMatrix<float>( m_center->x, m_center->y, m_center->z, m_extent );
}



void GMesh::updateBounds(gfloat3& low, gfloat3& high, GMatrixF& transform)
{
    if(m_low && m_high)
    {   
        gfloat3 mlow(*m_low) ; 
        gfloat3 mhigh(*m_high) ; 

        mlow  *= transform ; 
        mhigh *= transform ; 

        low.x = std::min( low.x, mlow.x);
        low.y = std::min( low.y, mlow.y);
        low.z = std::min( low.z, mlow.z);

        high.x = std::max( high.x, mhigh.x);
        high.y = std::max( high.y, mhigh.y);
        high.z = std::max( high.z, mhigh.z);
   }
}



gfloat3* GMesh::getTransformedVertices(GMatrixF& transform )
{
     gfloat3* vertices = new gfloat3[m_num_vertices];
     for(unsigned int i = 0; i < m_num_vertices; i++)
     {  
         vertices[i].x = m_vertices[i].x ;   
         vertices[i].y = m_vertices[i].y ;   
         vertices[i].z = m_vertices[i].z ;   

         vertices[i] *= transform ;
     }   
     return vertices ;
}




