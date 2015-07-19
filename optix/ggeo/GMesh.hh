#pragma once

#include "GVector.hh"
#include "GMatrix.hh"
#include "GDrawable.hh"

#include <vector>
#include <string>


class GBuffer ; 

//
// A comment claims are relatively few GMesh 
// which get reused with different transforms
// meaning that should put keep node specifics
// in the GNode and not in GMesh 
//
// Is this true ?
//
//     YES, the use of boundaries and nodes lists 
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

      static const char* vertices ; 
      static const char* normals ; 
      static const char* colors ; 
      static const char* texcoords ; 
      static const char* indices ; 
      static const char* nodes ; 
      static const char* boundaries ; 
      static const char* sensors ; 

      static const char* center_extent ; 
      static const char* transforms ; 
      static const char* meshes ; 

      GMesh(GMesh* other); // stealing copy ctor
      GMesh(unsigned int index, 
            gfloat3* vertices, unsigned int num_vertices, 
            guint3*  faces,     unsigned int num_faces,  
            gfloat3* normals, gfloat2* texcoords );
      virtual ~GMesh();

  public:
      void Summary(const char* msg="GMesh::Summary");
      void Dump(const char* msg="GMesh::Dump", unsigned int nmax=10);
  public:
      void setName(const char* name);
      const char* getName();
  public:
      gfloat3* getLow();
      gfloat3* getHigh();
  public:
      gfloat3* getCenter();  // TODO: move all users to CenterExtent
      gfloat4  getCenterExtent(unsigned int index);
      float* getTransform(unsigned int index);
      gfloat3* getDimensions();
      GMatrix<float>* getModelToWorld();

  public:
      unsigned int   getIndex();
      unsigned int   getNumVertices();
      unsigned int   getNumColors();
      unsigned int   getNumFaces();
      unsigned int   getNumSolids();
      unsigned int   getNumSolidsSelected();

  public:
      gfloat3*       getVertices();
      gfloat3*       getNormals();
      gfloat3*       getColors();
      gfloat2*       getTexcoords();
      guint3*        getFaces();

      bool hasTexcoords();

  public:
      // methods supporting save/load from file
      static GMesh* load(const char* dir);  
      void loadBuffers(const char* dir);
      void save(const char* dir);
      GBuffer* getBuffer(const char* name);
      void setBuffer(const char* name, GBuffer* buffer);
      bool isIntBuffer(const char* name);
      bool isUIntBuffer(const char* name);
      bool isFloatBuffer(const char* name);
      static void nameConstituents(std::vector<std::string>& names);
      void saveBuffer(const char* path, const char* name, GBuffer* buffer);
      void loadBuffer(const char* path, const char* name);

  public:
      void setVerticesBuffer(GBuffer* buffer);
      void setNormalsBuffer(GBuffer* buffer);
      void setColorsBuffer(GBuffer* buffer);
      void setTexcoordsBuffer(GBuffer* buffer);

      void setIndicesBuffer(GBuffer* buffer);
      void setNodesBuffer(GBuffer* buffer);
      void setBoundariesBuffer(GBuffer* buffer);
      void setSensorsBuffer(GBuffer* buffer);

      void setCenterExtentBuffer(GBuffer* buffer);
      void setTransformsBuffer(GBuffer* buffer);
      void setMeshesBuffer(GBuffer* buffer);

  public:
      // Buffer access for GDrawable protocol
      GBuffer* getVerticesBuffer();
      GBuffer* getNormalsBuffer();
      GBuffer* getColorsBuffer();
      GBuffer* getTexcoordsBuffer();
      GBuffer* getIndicesBuffer();
      GBuffer* getModelToWorldBuffer();
      GBuffer* getCenterExtentBuffer();
      GBuffer* getTransformsBuffer();
      GBuffer* getMeshesBuffer();

      float  getExtent();
      float* getModelToWorldPtr(unsigned int index);
      unsigned int findContainer(gfloat3 p);

  ///////// for use from subclass  /////////////////////////////////////
  public:
      virtual void setNodes(unsigned int* nodes);
      virtual void setBoundaries(unsigned int* boundaries);
      virtual void setSensors(   unsigned int* sensors);
  public:
      virtual unsigned int*  getNodes();
      virtual unsigned int*  getBoundaries();
      virtual unsigned int*  getSensors();

      virtual GBuffer* getNodesBuffer();
      virtual GBuffer* getBoundariesBuffer();
      virtual GBuffer* getSensorsBuffer();

      virtual std::vector<unsigned int>& getDistinctBoundaries();
      std::vector<std::string>& getNames();

  private:
      virtual void updateDistinctBoundaries();

  protected:
      unsigned int* m_nodes ; 
      unsigned int* m_boundaries ; 
      unsigned int* m_sensors ; 

  private: 
      std::vector<unsigned int> m_distinct_boundaries ;
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
      void setTransforms(float* transforms);
      void setMeshes(unsigned int* meshes);

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

  public:
      // used from GMergedMesh
      void setNumSolids(unsigned int numSolids);
  protected:
      unsigned int    m_index ;

      unsigned int    m_num_vertices ;
      unsigned int    m_num_colors ;
      unsigned int    m_num_faces ;
      unsigned int    m_num_solids  ;         // used from GMergedMesh subclass
      unsigned int    m_num_solids_selected  ;
 
      gfloat3*        m_vertices ;
      gfloat3*        m_normals ;
      gfloat3*        m_colors ;
      gfloat2*        m_texcoords ;
      guint3*         m_faces ;

      // TODO: get rid of these
      gfloat3*        m_low ;
      gfloat3*        m_high ;
      gfloat3*        m_dimensions ;
      gfloat3*        m_center ;
      float           m_extent ; 

      gfloat4*        m_center_extent ;
      float*          m_transforms ; 
      unsigned int*   m_meshes ; 

      GMatrix<float>* m_model_to_world ;  // does this make sense to be here ? for "unplaced" shape GMesh
      std::vector<std::string> m_names ;  // constituents with persistable buffers 
      const char*   m_name ; 

  private:
      GBuffer* m_vertices_buffer ;
      GBuffer* m_normals_buffer ;
      GBuffer* m_colors_buffer ;
      GBuffer* m_texcoords_buffer ;
      GBuffer* m_indices_buffer ;  // aka faces
      GBuffer* m_center_extent_buffer ;  
      GBuffer* m_nodes_buffer ;
      GBuffer* m_boundaries_buffer ;
      GBuffer* m_sensors_buffer ;
      GBuffer* m_transforms_buffer ;
      GBuffer* m_meshes_buffer ;


};




inline GMesh::~GMesh()
{
}

inline void GMesh::setName(const char* name)
{
     m_name = strdup(name);
}  
inline const char* GMesh::getName()
{
     return m_name ; 
}

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
inline unsigned int GMesh::getNumSolids()
{
    return m_num_solids ; 
}
inline unsigned int GMesh::getNumSolidsSelected()
{
    return m_num_solids_selected ; 
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


// index is used from subclass
inline gfloat4 GMesh::getCenterExtent(unsigned int index)
{
    return m_center_extent[index] ;
}
inline float* GMesh::getTransform(unsigned int index)
{
    return m_transforms + index*16  ;
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
inline unsigned int* GMesh::getBoundaries()
{
    return m_boundaries ;
}
inline unsigned int* GMesh::getSensors()
{
    return m_sensors ;
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
inline GBuffer*  GMesh::getCenterExtentBuffer()
{
    return m_center_extent_buffer ;
}
inline GBuffer*  GMesh::getTransformsBuffer()
{
    return m_transforms_buffer ;
}
inline GBuffer*  GMesh::getMeshesBuffer()
{
    return m_meshes_buffer ;
}
inline GBuffer*  GMesh::getIndicesBuffer()
{
    return m_indices_buffer ;
}
inline GBuffer*  GMesh::getNodesBuffer()
{
    return m_nodes_buffer ;
}
inline GBuffer*  GMesh::getBoundariesBuffer()
{
    return m_boundaries_buffer ;
}
inline GBuffer*  GMesh::getSensorsBuffer()
{
    return m_sensors_buffer ;
}



