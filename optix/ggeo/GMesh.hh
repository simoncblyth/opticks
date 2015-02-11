#ifndef GMESH_H
#define GMESH_H

#include "GVector.hh"

class GMesh {
  public:
      GMesh(gfloat3* vertices, unsigned int num_vertices, guint3* faces, unsigned int num_faces);
      virtual ~GMesh();

  private:
      unsigned int m_num_vertices ;
      unsigned int m_num_faces ; 
      gfloat3* m_vertices ;
      guint3*  m_faces ;

};

#endif
