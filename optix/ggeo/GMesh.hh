#ifndef GMESH_H
#define GMESH_H

#include "GVector.hh"
#include "GMatrix.hh"


class GMesh {
  public:
      GMesh(unsigned int index, gfloat3* vertices, unsigned int num_vertices, guint3* faces, unsigned int num_faces);
      virtual ~GMesh();

  public:
      void Summary(const char* msg="GMesh::Summary");
      unsigned int getIndex();
      gfloat3* getLow();
      gfloat3* getHigh();

  public:
      unsigned int getNumVertices();
      unsigned int getNumFaces();
      gfloat3* getVertices();
      guint3*  getFaces();

  public:
      void setLow(gfloat3* low);
      void setHigh(gfloat3* high);
      void setVertices(gfloat3* vertices);
      void setFaces(guint3* faces);
 
  public:
       gfloat3* getTransformedVertices(GMatrixF& transform );

  public:
      void updateBounds();
      void updateBounds(gfloat3& low, gfloat3& high, GMatrixF& transform);

  protected:
      unsigned int m_index ;
      unsigned int m_num_vertices ;
      unsigned int m_num_faces ; 
      gfloat3* m_vertices ;
      guint3*  m_faces ;
      gfloat3* m_low ;
      gfloat3* m_high ;


};

#endif
