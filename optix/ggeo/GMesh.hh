#ifndef GMESH_H
#define GMESH_H

#include "GVector.hh"
#include "GMatrix.hh"
#include "GDrawable.hh"

class GBuffer ; 

class GMesh : public GDrawable {
  public:
      GMesh(GMesh* other); // stealing copy ctor
      GMesh(unsigned int index, 
            gfloat3* vertices, unsigned int num_vertices, 
            guint3*  faces,     unsigned int num_faces,  
            gfloat3* normals, gfloat2* texcoords );
      virtual ~GMesh();

  public:
      void Summary(const char* msg="GMesh::Summary");
      void Dump(const char* msg="GMesh::Dump", unsigned int nmax=10);
      unsigned int getIndex();
      gfloat3* getLow();
      gfloat3* getHigh();
  public:
      gfloat3* getCenter();
      gfloat3* getDimensions();
      GMatrix<float>* getModelToWorld();

  public:
      unsigned int getNumVertices();
      unsigned int getNumColors();
      unsigned int getNumFaces();
      gfloat3* getVertices();
      gfloat3* getNormals();
      gfloat3* getColors();
      gfloat2* getTexcoords();
      guint3*  getFaces();

      bool hasTexcoords();

  public:
      // Buffer access for GDrawable protocol
      GBuffer* getVerticesBuffer();
      GBuffer* getNormalsBuffer();
      GBuffer* getColorsBuffer();
      GBuffer* getTexcoordsBuffer();
      GBuffer* getIndicesBuffer();
      GBuffer* getModelToWorldBuffer();
      float getExtent();

  public:
      void setLow(gfloat3* low);
      void setHigh(gfloat3* high);
      void setVertices(gfloat3* vertices);
      void setNormals(gfloat3* normals);
      void setColors(gfloat3* colors);
      void setTexcoords(gfloat2* texcoords);
      void setFaces(guint3* faces);

  public:
      void setNumColors(unsigned int num_colors);
      void setColor(float r, float g, float b);
 
  public:
       gfloat3* getTransformedVertices(GMatrixF& transform );
       gfloat3* getTransformedNormals(GMatrixF& transform );

  public:
      void updateBounds();
      void updateBounds(gfloat3& low, gfloat3& high, GMatrixF& transform);

  protected:
      unsigned int m_index ;
      unsigned int m_num_vertices ;
      unsigned int m_num_colors ;
      unsigned int m_num_faces ; 
      gfloat3* m_vertices ;
      gfloat3* m_normals ;
      gfloat3* m_colors ;
      gfloat2* m_texcoords ;
      guint3*  m_faces ;
      gfloat3* m_low ;
      gfloat3* m_high ;
      gfloat3* m_dimensions ;
      gfloat3* m_center ;
      float    m_extent ; 
      GMatrix<float>* m_model_to_world ; 

  private:
      GBuffer* m_vertices_buffer ;
      GBuffer* m_normals_buffer ;
      GBuffer* m_colors_buffer ;
      GBuffer* m_texcoords_buffer ;
      GBuffer* m_indices_buffer ;
      GBuffer* m_model_to_world_buffer ;


};

#endif
