#pragma once

struct NSlice ; 
template <typename T> class NPY ;
#include "NGLM.hpp"

class NPYBase ; 
class GParts ; 
class NCSG ; 


//struct nbbox ; 

template <typename T> class GMatrix ; 

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
#include "GDrawable.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GMesh : public GDrawable {
      friend class GMaker ;
      friend class GMergedMesh  ;
      friend class GMeshFixer  ;
      friend class GBBoxMesh ;
      friend class Texture ;
      friend class Demo;
     // TODO: too many friends, suggests need to improve isolation
  public:
      static int g_instance_count ; 

      // per-vertex
      static const char* vertices_ ; 
      static const char* normals_ ; 
      static const char* colors_ ; 
      static const char* texcoords_ ; 

      // per-face
      static const char* indices_ ; 

      static const char* nodes_ ; 
      static const char* boundaries_ ; 
      static const char* sensors_ ; 

      // per-solid (used from the composite GMergedMesh)
      static const char* center_extent_ ; 
      static const char* bbox_ ; 
      static const char* transforms_ ;    
      static const char* meshes_ ;        // mesh indices
      static const char* nodeinfo_ ;      // nface,nvert,?,? per solid : allowing solid selection beyond the geocache

      static const char* identity_ ;      // guint4: node, mesh, boundary, sensor : aiming to replace uint nodes/boundaries/sensors/meshes

      // per instance global transforms of repeated geometry 
      static const char* itransforms_ ;    
      static const char* iidentity_ ;     // guint4: node, mesh, boundary, sensor
      static const char* aiidentity_ ;    

      // composited GMergedMesh eg for LOD levels 
      static const char* components_ ;    


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
      void setVerbosity(unsigned verbosity); 
      unsigned getVerbosity() const ;
  private:
      void init(gfloat3* vertices, guint3* faces, gfloat3* normals, gfloat2* texcoords);
  public:
      GMesh* makeDedupedCopy();

      std::string desc() const ;
      void Summary(const char* msg="GMesh::Summary") const ;
      void dump(const char* msg="GMesh::dump", unsigned int nmax=10) const ;
      void dumpNormals(const char* msg="GMesh::dumpNormals", unsigned int nmax=10) const ;
  public:
      void setIndex(unsigned int index);
      void setName(const char* name);
      void setCSG(const NCSG* csg);
      void setAlt(const GMesh* alt);
    

      void setGeoCode(char geocode);
      void setInstanceSlice(NSlice* slice);
      void setFaceSlice(NSlice* slice);
      void setPartSlice(NSlice* slice);
      void setVersion(const char* version);

      const char* getName() const ;
      const char* getShortName() const ;
      const char* getVersion() const ;
      char getGeoCode() const ;
      const NCSG* getCSG() const ; 
      const GMesh* getAlt() const ; 

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
      gfloat4  getCenterExtent(unsigned int index) const ;
      //glm::vec4 getCenterExtent(unsigned int index);


      gbbox    getBBox(unsigned int index) const ;
      gbbox*   getBBoxPtr();

      float* getTransform(unsigned int index);
      float* getITransform(unsigned int index);
      gfloat3* getDimensions();
      GMatrix<float>* getModelToWorld();

  public:
      bool     isEmpty() const ;
      unsigned getIndex() const ;
      unsigned getNumVertices() const ;
      //unsigned getNumColors();
      unsigned getNumFaces() const ;
      unsigned getNumSolidsSelected() const ;
      unsigned getNumSolids() const ;
      int      getNumComponents() const ;
      void     dumpComponents(const char* msg="GMesh::dumpComponents") const ;

  public:
      unsigned getNumTransforms() ;
      unsigned getNumITransforms() ;

  public:
      // debug
       void explodeZVertices(float zoffset, float zcut);

  public:
      gfloat3*       getVertices();
      gfloat3*       getNormals();
      gfloat3*       getColors();
      gfloat2*       getTexcoords();
      bool hasTexcoords() const ;
  public:
      guint3*        getFaces() const ;

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
      void setComponentsBuffer(NPY<unsigned>* buf);
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
      // for composited GMergedMesh, eg for LOD levels 
      NPY<unsigned>*     getComponentsBuffer();
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

      virtual guint4*        getNodeInfo() const ;
      virtual guint4         getNodeInfo(unsigned int index);

      virtual guint4*        getIdentity() const ;
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

  private:
      void setLow(gfloat3* low);
      void setHigh(gfloat3* high);

  private:
      void setVertices(gfloat3* vertices);
      void setNormals(gfloat3* normals);
      void setColors(gfloat3* colors);
      void setTexcoords(gfloat2* texcoords);
  private:
      void setFaces(guint3* faces);
  private:
      void setBBox(gbbox* bb);
      void setTransforms(float* transforms);
      void setMeshes(unsigned int* meshes);
      void setNodeInfo(guint4* nodeinfo);
      void setIdentity(guint4* identity);
      //void setIIdentity(guint4* iidentity);   size numInstances*numSolids, try operating via the buffer setting only

  private:
      void setCenterExtent(gfloat4* center_extent);  // canonically invoked by GMesh::allocate


  public:
      void setColor(float r, float g, float b);
 
  public:
       gfloat3* getTransformedVertices(GMatrixF& transform ) const ;
       gfloat3* getTransformedNormals(GMatrixF& transform ) const ;

  public:
      static gbbox*  findBBox(gfloat3* vertices, unsigned int num_vertices);
      static gfloat4 findCenterExtentDeprecated(gfloat3* vertices, unsigned int num_vertices);
      void updateBounds();
      void updateBounds(gfloat3& low, gfloat3& high, GMatrixF& transform);

  public:
      // used from GMergedMesh
      void setNumVertices(unsigned int num_vertices);
      void setNumFaces(   unsigned int num_faces);
      void setNumSolids(unsigned int num_solids);
      void setComponent(const glm::uvec4& eidx, unsigned icomp );

  public:
      // analytic geometry standin for OptiX
      //void setParts(GParts* parts);
      //GParts* getParts();

  protected:
      unsigned     m_index ;

      unsigned     m_num_vertices ;
      unsigned     m_num_faces ;
      unsigned     m_num_solids  ;         // used from GMergedMesh subclass
      unsigned     m_num_solids_selected  ;
      unsigned     m_num_mergedmesh ;

      unsigned*    m_nodes ; 
      unsigned*    m_boundaries ; 
      unsigned*    m_sensors ; 
 
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
      NPY<unsigned>*     m_iidentity_buffer ;
      NPY<unsigned>*     m_aiidentity_buffer ; 
  protected:
      // Composited MM components, offset recording 
      NPY<unsigned>*     m_components_buffer ;   
  private:
      // transients
      GBuffer* m_facerepeated_identity_buffer ;
      GBuffer* m_facerepeated_iidentity_buffer ;
      GBuffer* m_analytic_geometry_buffer ; 

      //GParts*      m_parts ; 
      const NCSG*    m_csg ; 
      const GMesh*   m_alt ; 

      unsigned int   m_verbosity ; 

};


#include "GGEO_TAIL.hh"


