#ifndef GMESH_H
#define GMESH_H

#include "GVector.hh"

class GMesh {
  public:
      GMesh(unsigned int index, gfloat3* vertices, unsigned int num_vertices, guint3* faces, unsigned int num_faces);
      virtual ~GMesh();

  public:
      void Summary(const char* msg="GMesh::Summary");
      unsigned int getIndex();
      gfloat3* getLow();
      gfloat3* getHigh();

  private:
      void updateBounds();

  private:
      unsigned int m_index ;
      unsigned int m_num_vertices ;
      unsigned int m_num_faces ; 
      gfloat3* m_vertices ;
      guint3*  m_faces ;
      gfloat3* m_low ;
      gfloat3* m_high ;


};

#endif
