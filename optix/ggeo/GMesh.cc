#include "GMesh.hh"
#include "stdio.h"
#include <algorithm>


GMesh::GMesh(GMesh* other) 
     :
     m_index(other->getIndex()),
     m_vertices(other->getVertices()),
     m_num_vertices(other->getNumVertices()),
     m_faces(other->getFaces()),
     m_num_faces(other->getNumFaces()),
     m_colors(other->getColors()),
     m_num_colors(other->getNumColors())
{
   updateBounds();
}

GMesh::GMesh(unsigned int index, gfloat3* vertices, unsigned int num_vertices, guint3* faces, unsigned int num_faces) 
      :
      m_index(index),
      m_vertices(vertices),
      m_num_vertices(num_vertices), 
      m_faces(faces),
      m_num_faces(num_faces),
      m_low(NULL),
      m_high(NULL),
      m_num_colors(num_vertices) 
{
   // not yet taking ownership, depends on continued existance of data source 
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


gfloat3* GMesh::getLow()
{
    return m_low ;
}
gfloat3* GMesh::getHigh()
{
    return m_high ;
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


void GMesh::setVertices(gfloat3* vertices)
{
    m_vertices = vertices ;
}
void GMesh::setFaces(guint3* faces)
{
    m_faces = faces ;
}
void GMesh::setColors(gfloat3* colors)
{
    m_colors = colors ;
}



void GMesh::setLow(gfloat3* low)
{
    m_low = low ;
}
void GMesh::setHigh(gfloat3* high)
{
    m_high = high ;
}


void GMesh::setNumColors(unsigned int num_colors)
{
   m_num_colors = num_colors ;
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



