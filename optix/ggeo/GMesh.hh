#ifndef GMESH_H
#define GMESH_H

#include "GVector.hh"
#include "GMatrix.hh"
#include "GDrawable.hh"

class GBuffer ; 

//
// A comment claims are relatively few GMesh 
// which get reused with different transforms
// meaning that should put keep node specifics
// in the GNode and not in GMesh 
//
// Is this true ?
//
//     YES, the use of substances and nodes lists 
//     within the abstract unplaced shape GMesh 
//     does not make sense...
//
//     These should really be moved to the 
//     GMergedMesh subclass, but as a GDrawable 
//     related expedient are keeping them in here 
//     for now.
// 

class GMesh : public GDrawable {
  public:
      static int g_instance_count ; 

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
      gfloat3* getCenter();  // TODO: move all users to CenterExtent
      gfloat4  getCenterExtent(unsigned int index);
      gfloat3* getDimensions();
      GMatrix<float>* getModelToWorld();

  public:
      unsigned int   getNumVertices();
      unsigned int   getNumColors();
      unsigned int   getNumFaces();
      gfloat3*       getVertices();
      gfloat3*       getNormals();
      gfloat3*       getColors();
      gfloat2*       getTexcoords();
      guint3*        getFaces();

      bool hasTexcoords();

  public:
      // Buffer access for GDrawable protocol
      GBuffer* getVerticesBuffer();
      GBuffer* getNormalsBuffer();
      GBuffer* getColorsBuffer();
      GBuffer* getTexcoordsBuffer();
      GBuffer* getIndicesBuffer();
      GBuffer* getModelToWorldBuffer();
      GBuffer* getWavelengthBuffer();
      float  getExtent();
      float* getModelToWorldPtr(unsigned int index);

  ///////// expedients /////////////////////////////////////
  public:
      virtual void setNodes(unsigned int* nodes);
      virtual void setSubstances(unsigned int* substances);
  public:
      virtual unsigned int*  getNodes();
      virtual unsigned int*  getSubstances();
      virtual GBuffer* getNodesBuffer();
      virtual GBuffer* getSubstancesBuffer();
      virtual std::vector<unsigned int>& getDistinctSubstances();
  private:
      virtual void updateDistinctSubstances();

  protected:
      unsigned int* m_nodes ; 
      unsigned int* m_substances ; 
      GBuffer* m_wavelength_buffer ;

  private:
      GBuffer* m_nodes_buffer ;
      GBuffer* m_substances_buffer ;

  private: 
      std::vector<unsigned int> m_distinct_substances ;
  //////////////////////////////////////////////////////////////// 

  public:
      void setLow(gfloat3* low);
      void setHigh(gfloat3* high);
      void setVertices(gfloat3* vertices);
      void setNormals(gfloat3* normals);
      void setColors(gfloat3* colors);
      void setTexcoords(gfloat2* texcoords);
      void setFaces(guint3* faces);
      void setCenterExtent(gfloat4* center_extent);
      void setWavelengthBuffer(GBuffer* wavelength_buffer);

  public:
      void setNumColors(unsigned int num_colors);
      void setColor(float r, float g, float b);
 
  public:
       gfloat3* getTransformedVertices(GMatrixF& transform );
       gfloat3* getTransformedNormals(GMatrixF& transform );

  public:
      static gfloat4 findCenterExtent(gfloat3* vertices, unsigned int num_vertices);
      void updateBounds();
      void updateBounds(gfloat3& low, gfloat3& high, GMatrixF& transform);

  protected:
      unsigned int    m_index ;
      unsigned int    m_num_vertices ;
      unsigned int    m_num_colors ;
      unsigned int    m_num_faces ; 
      gfloat3*        m_vertices ;
      gfloat3*        m_normals ;
      gfloat3*        m_colors ;
      gfloat2*        m_texcoords ;
      guint3*         m_faces ;
      gfloat3*        m_low ;
      gfloat3*        m_high ;
      gfloat3*        m_dimensions ;
      gfloat3*        m_center ;
      float           m_extent ; 
      gfloat4*        m_center_extent ;
      GMatrix<float>* m_model_to_world ;  // does this make sense to be here ? for "unplaced" shape GMesh

  private:
      GBuffer* m_vertices_buffer ;
      GBuffer* m_normals_buffer ;
      GBuffer* m_colors_buffer ;
      GBuffer* m_texcoords_buffer ;
      GBuffer* m_indices_buffer ;  // aka faces


};


inline unsigned int GMesh::getIndex()
{
    return m_index ; 
}
inline unsigned int GMesh::getNumVertices()
{
    return m_num_vertices ; 
}
inline unsigned int GMesh::getNumColors()
{
    return m_num_colors ;   
}
inline unsigned int GMesh::getNumFaces()
{
    return m_num_faces ; 
}


inline void GMesh::setNumColors(unsigned int num_colors)
{
   m_num_colors = num_colors ;
}
inline void GMesh::setLow(gfloat3* low)
{
    m_low = low ;
}
inline void GMesh::setHigh(gfloat3* high)
{
    m_high = high ;
}
inline bool GMesh::hasTexcoords()
{
    return m_texcoords != NULL ;
}







inline gfloat3* GMesh::getLow()
{
    return m_low ;
}
inline gfloat3* GMesh::getHigh()
{
    return m_high ;
}
inline gfloat3* GMesh::getDimensions()
{
    return m_dimensions ; 
}

inline GMatrix<float>* GMesh::getModelToWorld()
{
    return m_model_to_world ; 
}





inline gfloat3* GMesh::getVertices()
{
    return m_vertices ;
}
inline gfloat3* GMesh::getNormals()
{
    return m_normals ;
}

inline gfloat3* GMesh::getColors()
{
    return m_colors ;
}
inline gfloat2* GMesh::getTexcoords()
{
    return m_texcoords ;
}


inline guint3*  GMesh::getFaces()
{
    return m_faces ;
}

inline gfloat4 GMesh::getCenterExtent(unsigned int index)
{
     return m_center_extent[0] ;   // index is used in subclass
}
inline void GMesh::setCenterExtent(gfloat4* center_extent)  // used from GMergedMesh
{
     m_center_extent = center_extent ;  
}





inline float GMesh::getExtent()
{
     return m_extent ;  
}



inline GBuffer*  GMesh::getModelToWorldBuffer()
{
    return (GBuffer*)m_model_to_world ;
}

inline float* GMesh::getModelToWorldPtr(unsigned int index)
{
     return (float*)getModelToWorldBuffer()->getPointer() ; 
}



inline unsigned int* GMesh::getNodes()   // CAUTION ONLY MAKES SENSE FROM GMergedMesh SUBCLASS 
{
    return m_nodes ;
}
inline unsigned int* GMesh::getSubstances()
{
    return m_substances ;
}
inline GBuffer*  GMesh::getWavelengthBuffer()
{
    return m_wavelength_buffer ;
}
inline void  GMesh::setWavelengthBuffer(GBuffer* wavelength_buffer)
{
    m_wavelength_buffer = wavelength_buffer ;
}





inline GBuffer* GMesh::getVerticesBuffer()
{
    return m_vertices_buffer ;
}
inline GBuffer* GMesh::getNormalsBuffer()
{
    return m_normals_buffer ;
}



inline GBuffer* GMesh::getColorsBuffer()
{
    return m_colors_buffer ;
}
inline GBuffer* GMesh::getTexcoordsBuffer()
{
    return m_texcoords_buffer ;
}



inline GBuffer*  GMesh::getIndicesBuffer()
{
    return m_indices_buffer ;
}
inline GBuffer*  GMesh::getNodesBuffer()
{
    return m_nodes_buffer ;
}
inline GBuffer*  GMesh::getSubstancesBuffer()
{
    return m_substances_buffer ;
}










#endif
