#include "GMesh.hh"
#include "stdio.h"

GMesh::GMesh(gfloat3* vertices, unsigned int num_vertices, guint3* faces, unsigned int num_faces) 
      :
      m_vertices(vertices),
      m_num_vertices(num_vertices), 
      m_faces(faces),
      m_num_faces(num_faces)  
{
   // not yet taking ownership, depends on continued existance of data source 
}

GMesh::~GMesh()
{
}

void GMesh::Summary(const char* msg)
{
   printf("%s vx %u fc %u \n", msg, m_num_vertices, m_num_faces );
}



