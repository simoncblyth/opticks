#pragma once

struct NSlice ; 
template <typename T> class NPY ;
class NPYBase ; 
class GParts ; 


#include "GMatrix.hh"
#include "GDrawable.hh"
#include "GVector.hh"

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
      contains per-solid nface, nvert, nodeindex, parent nodeindex
      nface counts per-solid allowing solid selection even from 
      merged meshes by accessing the correct ranges  

      ::

          In [1]: np.load("nodeinfo.npy")
          Out[1]: 
          array([[ 720,  362, 3199, 3155],
                 [ 672,  338, 3200, 3199],
                 [ 960,  482, 3201, 3200],
                 [ 480,  242, 3202, 3200],
                 [  96,   50, 3203, 3200]], dtype=uint32)


*identity*
      per-solid identity info, nodeIndex, meshIndex, boundaryIndex, sensorSurfaceIndex
      see GSolid::getIdentity
       
      ::

          In [2]: np.load("identity.npy")
          Out[2]: 
          array([[3199,   47,   19,    0],
                 [3200,   46,   20,    0],
                 [3201,   43,   21,    3],
                 [3202,   44,    1,    0],
                 [3203,   45,    1,    0]], dtype=uint32)





Indices relation to nodeinfo
-------------------------------

::

    In [3]: i = np.load("indices.npy")

    In [12]: i.reshape(-1,3)
    Out[12]: 
    array([[   0,    1,    2],
           [   0,    2,    3],
           [   0,    3,    4],
           ..., 
           [1466, 1473, 1468],
           [1468, 1473, 1470],
           [1470, 1473, 1425]], dtype=int32)

    In [13]: i.reshape(-1,3).shape
    Out[13]: (2928, 3)


    In [6]: ni = np.load("nodeinfo.npy")

    In [7]: ni
    Out[7]: 
    array([[ 720,  362, 3199, 3155],
           [ 672,  338, 3200, 3199],
           [ 960,  482, 3201, 3200],
           [ 480,  242, 3202, 3200],
           [  96,   50, 3203, 3200]], dtype=uint32)

    In [9]: ni[:,0].sum()
    Out[9]: 2928


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
      static const char* transforms ;    
      static const char* meshes ;        // mesh indices
      static const char* nodeinfo ;      // nface,nvert,?,? per solid : allowing solid selection beyond the geocache

      static const char* identity ;      // guint4: node, mesh, boundary, sensor : aiming to replace uint nodes/boundaries/sensors/meshes

      // per instance global transforms of repeated geometry 
      static const char* itransforms ;    
      static const char* iidentity ;     // guint4: node, mesh, boundary, sensor
      static const char* aiidentity ;    

      static GMesh* make_spherelocal_mesh(NPY<float>* triangles, unsigned int meshindex=0);  
      static GMesh* make_mesh(NPY<float>* triangles, unsigned int meshindex=0);

      GMesh(unsigned int index=0, 
            gfloat3* vertices=NULL, unsigned int num_vertices=0, 
            guint3*  faces=NULL,    unsigned int num_faces=0,  
            gfloat3* normals=NULL, 
            gfloat2* texcoords=NULL );

      void allocate();  // must first have set numVertices, numFaces, numSolids
      void deallocate(); 
      virtual ~GMesh();
      void setVerbosity(unsigned int verbosity); 
      unsigned int getVerbosity();
  private:
      void init(gfloat3* vertices, guint3* faces, gfloat3* normals, gfloat2* texcoords);
  public:
      GMesh* makeDedupedCopy();

      void Summary(const char* msg="GMesh::Summary");
      void dump(const char* msg="GMesh::dump", unsigned int nmax=10);
      void dumpNormals(const char* msg="GMesh::dumpNormals", unsigned int nmax=10);
  public:
      void setIndex(unsigned int index);
      void setName(const char* name);
      void setGeoCode(char geocode);
      void setInstanceSlice(NSlice* slice);
      void setFaceSlice(NSlice* slice);
      void setPartSlice(NSlice* slice);
      void setVersion(const char* version);

      const char* getName();
      const char* getShortName();
      const char* getVersion();
      char getGeoCode();
      NSlice* getInstanceSlice();
      NSlice* getFaceSlice();
      NSlice* getPartSlice();


  private:
      void findShortName(); 
  public:
      gfloat3* getLow();
      gfloat3* getHigh();
  public:
      gfloat3* getCenter();  // TODO: move all users to CenterExtent
      gfloat4  getCenterExtent(unsigned int index);


      gbbox    getBBox(unsigned int index);
      gbbox*   getBBoxPtr();
      float* getTransform(unsigned int index);
      float* getITransform(unsigned int index);
      gfloat3* getDimensions();
      GMatrix<float>* getModelToWorld();

  public:
      unsigned int   getIndex();
      unsigned int   getNumVertices();
      unsigned int   getNumColors();
      unsigned int   getNumFaces();
      unsigned int   getNumSolidsSelected();
      unsigned int   getNumSolids();
      unsigned int   getNumTransforms();

      unsigned int   getNumITransforms();

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

      // saves into basedir/typedir/instancedir, only basedir is required
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
      bool isNPYBuffer(const char* name);
      void loadNPYBuffer(const char* path, const char* name);
      void saveNPYBuffer(const char* path, const char* name);
      NPYBase* getNPYBuffer(const char* name);
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
      void setIdentityBuffer(GBuffer* buffer);
  public:
      void setITransformsBuffer(NPY<float>* buf);
      void setInstancedIdentityBuffer(NPY<unsigned int>* buf);
      void setAnalyticInstancedIdentityBuffer(NPY<unsigned int>* buf);
  public:
      bool hasTransformsBuffer(); 
      bool hasITransformsBuffer(); 
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
      GBuffer* getIdentityBuffer();
  public:
      // all instanced buffers created by GTreeCheck
      NPY<unsigned int>* getAnalyticInstancedIdentityBuffer();
      NPY<float>*        getITransformsBuffer();
      NPY<unsigned int>* getInstancedIdentityBuffer(); 

  public:
      float  getExtent();
      float* getModelToWorldPtr(unsigned int index);
      unsigned int findContainer(gfloat3 p);

  public:
      // transient buffers, not persisted : providing node level info in a face level buffer by repetition
      GBuffer* getFaceRepeatedInstancedIdentityBuffer(); 
      GBuffer* getFaceRepeatedIdentityBuffer(); 
      GBuffer* getAnalyticGeometryBuffer();
  private: 
      GBuffer* makeFaceRepeatedInstancedIdentityBuffer();
      GBuffer* makeFaceRepeatedIdentityBuffer();
      GBuffer* loadAnalyticGeometryBuffer(const char* path); 

  ///////// for use from subclass  /////////////////////////////////////
  public:
      virtual void setNodes(unsigned int* nodes);
      virtual void setBoundaries(unsigned int* boundaries);
      virtual void setSensors(   unsigned int* sensors);
  public:
      virtual unsigned int*  getNodes();
      virtual unsigned int*  getBoundaries();
      virtual unsigned int*  getSensors();

      virtual guint4*        getNodeInfo();
      virtual guint4         getNodeInfo(unsigned int index);

      virtual guint4*        getIdentity();
      virtual guint4         getIdentity(unsigned int index);

      virtual guint4*        getInstancedIdentity();
      virtual guint4         getInstancedIdentity(unsigned int index);

      virtual unsigned int*  getMeshIndice();
      virtual unsigned int   getMeshIndice(unsigned int index);


      virtual GBuffer* getNodesBuffer();
      virtual GBuffer* getBoundariesBuffer();
      virtual GBuffer* getSensorsBuffer();

      virtual std::vector<unsigned int>& getDistinctBoundaries();
      std::vector<std::string>& getNames();

  private:
      virtual void updateDistinctBoundaries();

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
      void setIdentity(guint4* identity);
      //void setIIdentity(guint4* iidentity);   size numInstances*numSolids, try operating via the buffer setting only

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

  public:
      // analytic geometry standin for OptiX
      void setParts(GParts* parts);
      GParts* getParts();

  protected:
      unsigned int    m_index ;

      unsigned int    m_num_vertices ;
      unsigned int    m_num_faces ;
      unsigned int    m_num_solids  ;         // used from GMergedMesh subclass
      unsigned int    m_num_solids_selected  ;

      unsigned int*   m_nodes ; 
      unsigned int*   m_boundaries ; 
      unsigned int*   m_sensors ; 
 
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
      float*          m_itransforms ; 
      unsigned int*   m_meshes ; 
      guint4*         m_nodeinfo ; 
      guint4*         m_identity ; 
      guint4*         m_iidentity ; 


      GMatrix<float>* m_model_to_world ;  // does this make sense to be here ? for "unplaced" shape GMesh
      std::vector<std::string> m_names ;  // constituents with persistable buffers 
      const char*   m_name ; 
      const char*   m_shortname ; 
      const char*   m_version ; 
      char          m_geocode ; 
      NSlice*       m_islice ; 
      NSlice*       m_fslice ; 
      NSlice*       m_pslice ; 

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
      GBuffer* m_identity_buffer ;
  private:
      // instancing related  buffers created by GTreeCheck 
      NPY<float>*        m_itransforms_buffer ;
      NPY<unsigned int>* m_iidentity_buffer ;
      NPY<unsigned int>* m_aiidentity_buffer ; 

  private:
      // transients
      GBuffer* m_facerepeated_identity_buffer ;
      GBuffer* m_facerepeated_iidentity_buffer ;
      GBuffer* m_analytic_geometry_buffer ; 

      GParts*      m_parts ; 

      unsigned int   m_verbosity ; 

};



inline GMesh::GMesh(unsigned int index, 
             gfloat3* vertices, 
             unsigned int num_vertices, 
             guint3* faces, 
             unsigned int num_faces, 
             gfloat3* normals, 
             gfloat2* texcoords
            ) 
        :
      GDrawable(),
      m_index(index),

      m_num_vertices(num_vertices), 
      m_num_faces(num_faces),
      m_num_solids(0),
      m_num_solids_selected(0),

      m_nodes(NULL),          
      m_boundaries(NULL),
      m_sensors(NULL),

      m_vertices(NULL),
      m_normals(NULL),
      m_colors(NULL),
      m_texcoords(NULL),
      m_faces(NULL),

      m_low(NULL),
      m_high(NULL),
      m_dimensions(NULL),
      m_center(NULL),
      m_extent(0.f),

      m_center_extent(NULL),
      m_bbox(NULL),
      m_transforms(NULL),
      m_itransforms(NULL),
      m_meshes(NULL),
      m_nodeinfo(NULL),
      m_identity(NULL),
      m_iidentity(NULL),

      m_model_to_world(NULL),
      m_name(NULL),
      m_shortname(NULL),
      m_version(NULL),
      m_geocode('T'),
      m_islice(NULL),
      m_fslice(NULL),
      m_pslice(NULL),

      m_vertices_buffer(NULL),
      m_normals_buffer(NULL),
      m_colors_buffer(NULL),
      m_texcoords_buffer(NULL),
      m_indices_buffer(NULL),
      m_center_extent_buffer(NULL),
      m_bbox_buffer(NULL),
      m_boundaries_buffer(NULL),
      m_sensors_buffer(NULL),
      m_transforms_buffer(NULL),
      m_meshes_buffer(NULL),
      m_nodeinfo_buffer(NULL),
      m_identity_buffer(NULL),

      m_itransforms_buffer(NULL),
      m_iidentity_buffer(NULL),
      m_aiidentity_buffer(NULL),

      m_facerepeated_identity_buffer(NULL),
      m_facerepeated_iidentity_buffer(NULL),
      m_analytic_geometry_buffer(NULL),

      m_parts(NULL),
      m_verbosity(0)
{
     init(vertices, faces, normals, texcoords);
}



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
    delete[] m_itransforms ;  
    delete[] m_meshes ;  
    delete[] m_nodeinfo ;  
    delete[] m_identity ;  
    delete[] m_iidentity ;  

    // NB buffers and the rest are very lightweight 
}


inline GMesh::~GMesh()
{
    deallocate();
}

inline void GMesh::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}

inline unsigned int GMesh::getVerbosity()
{
    return m_verbosity ; 
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
inline gbbox* GMesh::getBBoxPtr()
{
    return m_bbox ;
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



inline unsigned int* GMesh::getMeshIndice()  
{
    return m_meshes ;
}
inline unsigned int GMesh::getMeshIndice(unsigned int index)  
{
    return m_meshes[index] ;
}



inline guint4* GMesh::getNodeInfo()
{
    return m_nodeinfo ; 
}
inline guint4 GMesh::getNodeInfo(unsigned int index)
{
    return m_nodeinfo[index] ; 
}

inline guint4* GMesh::getIdentity()
{
    return m_identity ; 
}
inline guint4 GMesh::getIdentity(unsigned int index)
{
    return m_identity[index] ; 
}

inline guint4* GMesh::getInstancedIdentity()
{
    return m_iidentity ; 
}
inline guint4 GMesh::getInstancedIdentity(unsigned int index)
{
    return m_iidentity[index] ; 
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
inline NPY<float>*  GMesh::getITransformsBuffer()
{
    return m_itransforms_buffer ;
}



inline GBuffer*  GMesh::getMeshesBuffer()
{
    return m_meshes_buffer ;
}
inline GBuffer*  GMesh::getNodeInfoBuffer()
{
    return m_nodeinfo_buffer ;
}
inline GBuffer*  GMesh::getIdentityBuffer()
{
    return m_identity_buffer ;
}
inline NPY<unsigned int>*  GMesh::getInstancedIdentityBuffer()
{
    return m_iidentity_buffer ;
}
inline NPY<unsigned int>*  GMesh::getAnalyticInstancedIdentityBuffer()
{
    return m_aiidentity_buffer ;
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
inline bool GMesh::hasITransformsBuffer()
{
    return m_itransforms_buffer != NULL ; 
}







inline char GMesh::getGeoCode()
{
    return m_geocode ; 
}
inline void GMesh::setGeoCode(char geocode)
{
    m_geocode = geocode ; 
}


inline void GMesh::setInstanceSlice(NSlice* slice)
{
    m_islice = slice ; 
}
inline NSlice* GMesh::getInstanceSlice()
{
    return m_islice ; 
}


inline void GMesh::setFaceSlice(NSlice* slice)
{
    m_fslice = slice ; 
}
inline NSlice* GMesh::getFaceSlice()
{
    return m_fslice ; 
}

inline void GMesh::setPartSlice(NSlice* slice)
{
    m_pslice = slice ; 
}
inline NSlice* GMesh::getPartSlice()
{
    return m_pslice ; 
}



inline void GMesh::setParts(GParts* parts)
{
    m_parts = parts ; 
}
inline GParts* GMesh::getParts()
{
    return m_parts ; 
}







