#pragma once

#include "GVector.hh"
#include "GMatrix.hh"
#include "GDrawable.hh"

#include <vector>
#include <string>


class GBuffer ; 

/*

GMesh
======

GMesh are distinct geometrical shapes, **NOT** placed nodes/solids
--------------------------------------------------------------------

There are relatively few GMesh which get reused with different transforms
meaning that node specifics reside in the GNode, not in GMesh. 

For example the use of boundaries and nodes lists 
within the abstract unplaced shape GMesh 
does not make sense...


GMergedMesh isa GMesh that combines the data from many GMesh into one
-----------------------------------------------------------------------

Boundaries and nodes lists make sense at GMergedMesh level as there 
are only a few GMergedMesh instances for the entire geometry.


Vertex buffers in a GMesh
----------------------------------

*vertices*
      3*float*nvert

*normals*
      3*float*nvert

*colors*
      3*float*nvert

*texcoords*
      2*float*nvert


Face buffers in a GMesh (used from GMergedMesh)
-----------------------------------------------

*indices*
     3*uint*nface

*nodes*
     1*uint*nface

*boundaries*
     1*uint*nface

*sensors*
     1*uint*nface


Solid buffers in a GMesh (used from GMergedMesh)
-------------------------------------------------

*center_extent*

*bbox*

*transforms*

*meshes*

*nodeinfo*
      contains per-solid nface, nvert allowing solid selection even from 
      merged meshes by accessing the correct ranges  



*/

class GMesh : public GDrawable {
      friend class GMergedMesh  ;
  public:
      static int g_instance_count ; 

      // per-vertex
      static const char* vertices ; 
      static const char* normals ; 
      static const char* colors ; 
      static const char* texcoords ; 

      // per-face
      static const char* indices ; 
      static const char* nodes ; 
      static const char* boundaries ; 
      static const char* sensors ; 

      // per-solid (used from the composite GMergedMesh)
      static const char* center_extent ; 
      static const char* bbox ; 
      static const char* transforms ;    // not-used? 
      static const char* meshes ;        // mesh indices
      static const char* nodeinfo ;      // nface,nvert,?,? per solid : allowing solid selection beyond the geocache

      GMesh(GMesh* other); // stealing copy ctor
      GMesh(unsigned int index=0, 
            gfloat3* vertices=NULL, unsigned int num_vertices=0, 
            guint3*  faces=NULL,    unsigned int num_faces=0,  
            gfloat3* normals=NULL, 
            gfloat2* texcoords=NULL );

      void allocate();  // must first have set numVertices, numFaces, numSolids
      void deallocate(); 
      virtual ~GMesh();

  public:
      GMesh* makeDedupedCopy();

      void Summary(const char* msg="GMesh::Summary");
      void Dump(const char* msg="GMesh::Dump", unsigned int nmax=10);
  public:
      void setIndex(unsigned int index);
      void setName(const char* name);
      void setVersion(const char* version);
      const char* getName();
      const char* getShortName();
      const char* getVersion();
  private:
      void findShortName(); 
  public:
      gfloat3* getLow();
      gfloat3* getHigh();
  public:
      gfloat3* getCenter();  // TODO: move all users to CenterExtent
      gfloat4  getCenterExtent(unsigned int index);
      gbbox    getBBox(unsigned int index);
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
      // debug
       void explodeZVertices(float zoffset, float zcut);

  public:
      gfloat3*       getVertices();
      gfloat3*       getNormals();
      gfloat3*       getColors();
      gfloat2*       getTexcoords();
      bool hasTexcoords();
  public:
      guint3*        getFaces();

  public:
      static GMesh* load(const char* basedir, const char* typedir=NULL, const char* instancedir=NULL);
      static GMesh* load_deduped(const char* basedir, const char* typedir=NULL, const char* instancedir=NULL);
      void save(const char* basedir, const char* typedir=NULL, const char* instancedir=NULL);

  private:
      // methods supporting save/load from file
      std::string getVersionedBufferName(std::string& name);
      void loadBuffers(const char* dir);
      void saveBuffers(const char* dir);
      void saveBuffer(const char* path, const char* name, GBuffer* buffer);
      void loadBuffer(const char* path, const char* name);
      static void nameConstituents(std::vector<std::string>& names);
  private: 
      GBuffer* getBuffer(const char* name);
      void setBuffer(const char* name, GBuffer* buffer);
      bool isIntBuffer(const char* name);
      bool isUIntBuffer(const char* name);
      bool isFloatBuffer(const char* name);
  public:
      void setVerticesBuffer(GBuffer* buffer);
      void setNormalsBuffer(GBuffer* buffer);
      void setColorsBuffer(GBuffer* buffer);
      void setTexcoordsBuffer(GBuffer* buffer);

      void setIndicesBuffer(GBuffer* buffer);   // TODO: rename FacesBuffer
      void setNodesBuffer(GBuffer* buffer);     // TODO: consolidate the 3 into FaceInfoBuffer
      void setBoundariesBuffer(GBuffer* buffer);
      void setSensorsBuffer(GBuffer* buffer);

      void setCenterExtentBuffer(GBuffer* buffer);
      void setBBoxBuffer(GBuffer* buffer);
      void setTransformsBuffer(GBuffer* buffer);
      void setMeshesBuffer(GBuffer* buffer);
      void setNodeInfoBuffer(GBuffer* buffer);

  public:
      bool hasTransformsBuffer(); 
      unsigned int getNumTransforms();

  public:
      // Buffer access for GDrawable protocol
      GBuffer* getVerticesBuffer();
      GBuffer* getNormalsBuffer();
      GBuffer* getColorsBuffer();
      GBuffer* getTexcoordsBuffer();

      GBuffer* getIndicesBuffer();

      GBuffer* getModelToWorldBuffer();
      GBuffer* getCenterExtentBuffer();
      GBuffer* getBBoxBuffer();
      GBuffer* getTransformsBuffer();
      GBuffer* getMeshesBuffer();
      GBuffer* getNodeInfoBuffer();

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
      virtual guint4*  getNodeInfo();

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

  public:
      void setVertices(gfloat3* vertices);
      void setNormals(gfloat3* normals);
      void setColors(gfloat3* colors);
      void setTexcoords(gfloat2* texcoords);
  public:
      void setFaces(guint3* faces);
  public:
      void setCenterExtent(gfloat4* center_extent);
      void setBBox(gbbox* bb);
      void setTransforms(float* transforms);
      void setMeshes(unsigned int* meshes);
      void setNodeInfo(guint4* nodeinfo);

  public:
      void setColor(float r, float g, float b);
 
  public:
       gfloat3* getTransformedVertices(GMatrixF& transform );
       gfloat3* getTransformedNormals(GMatrixF& transform );

  public:
      static gbbox   findBBox(gfloat3* vertices, unsigned int num_vertices);
      static gfloat4 findCenterExtentDeprecated(gfloat3* vertices, unsigned int num_vertices);
      void updateBounds();
      void updateBounds(gfloat3& low, gfloat3& high, GMatrixF& transform);

  public:
      // used from GMergedMesh
      void setNumVertices(unsigned int num_vertices);
      void setNumFaces(   unsigned int num_faces);
      void setNumSolids(unsigned int num_solids);


  protected:
      unsigned int    m_index ;

      unsigned int    m_num_vertices ;
      //unsigned int    m_num_colors ;
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

      // per-solid/node 
      gfloat4*        m_center_extent ;
      gbbox*          m_bbox ;
      float*          m_transforms ; 
      unsigned int*   m_meshes ; 
      guint4*         m_nodeinfo ; 



      GMatrix<float>* m_model_to_world ;  // does this make sense to be here ? for "unplaced" shape GMesh
      std::vector<std::string> m_names ;  // constituents with persistable buffers 
      const char*   m_name ; 
      const char*   m_shortname ; 
      const char*   m_version ; 

  private:
      GBuffer* m_vertices_buffer ;
      GBuffer* m_normals_buffer ;
      GBuffer* m_colors_buffer ;
      GBuffer* m_texcoords_buffer ;

      GBuffer* m_indices_buffer ;  // aka faces

      GBuffer* m_center_extent_buffer ;  
      GBuffer* m_bbox_buffer ;  
      GBuffer* m_nodes_buffer ;
      GBuffer* m_boundaries_buffer ;
      GBuffer* m_sensors_buffer ;
      GBuffer* m_transforms_buffer ;
      GBuffer* m_meshes_buffer ;
      GBuffer* m_nodeinfo_buffer ;


};



inline void GMesh::deallocate()
{
    delete[] m_vertices ;  
    delete[] m_normals ;  
    delete[] m_colors ;  
    delete[] m_texcoords ;  
    delete[] m_faces ;  

    delete[] m_center_extent ;  
    delete[] m_bbox ;  
    delete[] m_transforms ;  
    delete[] m_meshes ;  
    delete[] m_nodeinfo ;  

    // NB buffers and the rest are very lightweight 
}


inline GMesh::~GMesh()
{
    deallocate();
}

inline void GMesh::setName(const char* name)
{
     m_name = name ? strdup(name) : NULL ;
     if(m_name) findShortName();
}  
inline const char* GMesh::getName()
{
     return m_name ; 
}
inline const char* GMesh::getShortName()
{
     return m_shortname ; 
}


inline void GMesh::setVersion(const char* version)
{
     m_version = version ? strdup(version) : NULL ;
}  
inline const char* GMesh::getVersion()
{
     return m_version ; 
}




inline unsigned int GMesh::getIndex()
{
    return m_index ; 
}
inline unsigned int GMesh::getNumVertices()
{
    return m_num_vertices ; 
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




inline void GMesh::setIndex(unsigned int index)
{
   m_index = index ;
}
inline void GMesh::setNumVertices(unsigned int num_vertices)
{
    m_num_vertices = num_vertices ; 
}
inline void GMesh::setNumFaces(unsigned int num_faces)
{
    m_num_faces = num_faces ; 
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
inline gbbox GMesh::getBBox(unsigned int index)
{
    return m_bbox[index] ;
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
inline guint4* GMesh::getNodeInfo()
{
    return m_nodeinfo ; 
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
inline GBuffer*  GMesh::getBBoxBuffer()
{
    return m_bbox_buffer ;
}
inline GBuffer*  GMesh::getTransformsBuffer()
{
    return m_transforms_buffer ;
}
inline GBuffer*  GMesh::getMeshesBuffer()
{
    return m_meshes_buffer ;
}
inline GBuffer*  GMesh::getNodeInfoBuffer()
{
    return m_nodeinfo_buffer ;
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

inline bool GMesh::hasTransformsBuffer()
{
    return m_transforms_buffer != NULL ; 
}




