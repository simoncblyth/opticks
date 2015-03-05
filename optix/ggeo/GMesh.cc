#include "GMesh.hh"
#include "stdio.h"
#include <algorithm>


GMesh::GMesh(unsigned int index, gfloat3* vertices, unsigned int num_vertices, guint3* faces, unsigned int num_faces) 
      :
      m_index(index),
      m_vertices(vertices),
      m_num_vertices(num_vertices), 
      m_faces(faces),
      m_num_faces(num_faces),
      m_low(NULL),
      m_high(NULL)  
{
   // not yet taking ownership, depends on continued existance of data source 
   updateBounds(); 
}


unsigned int GMesh::getIndex()
{
     return m_index ; 
}


GMesh::~GMesh()
{
}

void GMesh::Summary(const char* msg)
{
   printf("%s idx %u vx %u fc %u \nlow  %10.3f %10.3f %10.3f\nhigh %10.3f %10.3f %10.3f\n", 
      msg, 
      m_index, 
      m_num_vertices, 
      m_num_faces, 
      m_low->x,
      m_low->y,
      m_low->z,
      m_high->x,
      m_high->y,
      m_high->z
   );
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


