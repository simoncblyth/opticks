#pragma once

#include <map>
#include <vector>
#include <unordered_set>
#include <iterator>

#include <glm/fwd.hpp>
#include "plog/Severity.h"

// npy-
#include "NConfigurable.hpp"

class NLookup ; 
class NMeta ;
class TorchStepNPY ; 
class SLog ; 

// okc-
class Opticks ; 
class OpticksEvent ; 
class OpticksColors ; 
class OpticksFlags ; 
class OpticksResource ; 
class OpticksAttrSeq ; 
class Composition ; 

// ggeo-
#include "GVector.hh"
template <typename T> class GDomain ; 
template <typename T> class GPropertyMap ; 
template <typename T> class GProperty ; 

class GMesh ; 
class GVolume ; 
class GNode ; 
class GMaterial ; 
class GSkinSurface ; 
class GBorderSurface ; 

class GMeshLib ; 
class GNodeLib ; 
class GGeoLib ;
class GBndLib ;
class GMaterialLib ;
class GSurfaceLib ;
class GScintillatorLib ;
class GSourceLib ;
class GPmtLib ; 


class GInstancer ;
class GColorizer ; 

class GItemIndex ; 
class GItemList ; 
class GMergedMesh ;

// GLTF handling 
class GScene ; 

#include "GGeoBase.hh"

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/*
GGeo
=====

In the beginning GGeo was intended to be  a dumb substrate 
from which the geometry model is created eg by AssimpGGeo::convert 
However it grew to be somewhat monolithic.

When possible break pieces off the monolith.

Primary Constituents
----------------------

Opticks
Composition
GInstancer
NLookup
GMeshLib
GGeoLib
GNodeLib
   precache : holds GVolume
   persists pvnames, lvname

GBndLib
GMaterialLib
GSurfaceLib
GScintillatorLib
GSourceLib
GPmtLib

GColorizer
GScene


*/

class GGEO_API GGeo : public GGeoBase, public NConfigurable {
    public:
        friend class  X4PhysicalVolume ;  // X4PhysicalVolume::init needs afterConvertMaterial 
        friend class  AssimpGGeo ; 
        friend struct GSceneTest ; 
    public:
        static const plog::Severity LEVEL ; 
        static GGeo* GetInstance();  // statically provides the last instanciated GGeo instance
        static const char* CATHODE_MATERIAL ; 
    public:
        // see GGeoCfg.hh
        static const char* PICKFACE ;   
        static const char* PREFIX ;
    public:
        // GGeoBase interface : so not so easy to const-ify 

        GScintillatorLib* getScintillatorLib() const ; 
        GSourceLib*       getSourceLib() const ; 
        GSurfaceLib*      getSurfaceLib() const ;
        GMaterialLib*     getMaterialLib() const ;

        GBndLib*          getBndLib() const ; 
        GPmtLib*          getPmtLib() const ; 
        GGeoLib*          getGeoLib()  const ; 
        GNodeLib*         getNodeLib() const ;

        const char*       getIdentifier()  const ;
        GMergedMesh* getMergedMesh(unsigned int index) const ;

        // GGeoBase interace END
    public:
        // NConfigurable
        const char* getPrefix();
        void configure(const char* name, const char* value);
        std::vector<std::string> getTags();
        void set(const char* name, std::string& s);
        std::string get(const char* name);
    public:
        typedef int (*GLoaderImpFunctionPtr)(GGeo*);
        void setLoaderImp(GLoaderImpFunctionPtr imp);
        void setMeshVerbosity(unsigned int verbosity);
        unsigned int getMeshVerbosity() const ;
    public:
        typedef GMesh* (*GJoinImpFunctionPtr)(GMesh*, Opticks*);
        void setMeshJoinImp(GJoinImpFunctionPtr imp);
        void setMeshJoinCfg(const char* config);
        bool shouldMeshJoin(const GMesh* mesh);
        GMesh* invokeMeshJoin(GMesh* mesh);    // used from AssimpGGeo::convertMeshes immediately after GMesh birth and deduping
    public:
        typedef std::map<unsigned int, std::string> Index_t ;

    public:
        GGeo(Opticks* opticks, bool live=false); 
    public:
        const char* getIdPath();
        bool isValid() const ;
        bool isLive() const ;
    public:
        Composition* getComposition();
        void setComposition(Composition* composition);
    public:
        void loadGeometry(); 
        void loadFromCache();
        void loadFromG4DAE();  // AssimpGGeo::load
        void postDirectTranslation(); 
    private: 
        void loadAnalyticFromGLTF();
        void loadAnalyticFromCache();

        void afterConvertMaterials();
        //void createSurLib();
    public:
        // post-load setup
        void setupLookup();
        void setupColors();
        void setupTyp();
    public:
        // configureGeometry stage additions
    public:
        bool isPrepared() const ; 
        void prepare();  // prepare is needed before saving to file or GPU upload by oxrap.OGeo
    public:
        void close();
        void prepareMaterialLib();
        void prepareSurfaceLib();
        void prepareScintillatorLib();
        void prepareSourceLib();
        void prepareVolumes();   
        void prepareVertexColors();
    public:

        unsigned int getMaterialLine(const char* shortname);

   private:
        void init(); 
        //void loadMergedMeshes(const char* idpath);
        //void removeMergedMeshes(const char* idpath);
    public:
        void save();
        void saveAnalytic();
        void anaEvent(OpticksEvent* evt);
    private:
        //void saveMergedMeshes(const char* idpath);
    public:
        // pass thru to geolib
        GMergedMesh* makeMergedMesh(unsigned int index, GNode* base, GNode* root, unsigned verbosity );
        unsigned int getNumMergedMesh();
    public:
        // these are operational from cache
        // target 0 : all geometry of the mesh, >0 : specific volumes
        glm::vec4 getCenterExtent(unsigned int target, unsigned int merged_mesh_index=0u );
        void dumpTree(const char* msg="GGeo::dumpTree");  
        void dumpVolume(unsigned int index, const char* msg="GGeo::dumpVolume");  
        void dumpNodeInfo(unsigned int mmindex, const char* msg="GGeo::dumpNodeInfo" );
        void dumpStats(const char* msg="GGeo::dumpStats");

        // merged mesh buffer offsets and counts
        //
        //     .x  prior faces offset    
        //     .y  prior verts offset  
        //     .z  index faces count
        //     .w  index verts count
        //
        glm::ivec4 getNodeOffsetCount(unsigned int index);
        glm::vec4 getFaceCenterExtent(unsigned int face_index, unsigned int volume_index, unsigned int mergedmesh_index=0 );
        glm::vec4 getFaceRangeCenterExtent(unsigned int face_index0, unsigned int face_index1, unsigned int volume_index, unsigned int mergedmesh_index=0 );

    private:
        glm::mat4 getTransform(int index);  //TRYING TO MOVE TO HUB 
    public:
        bool isLoaded();

    public:
        // via GNodeLib
        void add(GVolume*    volume);

        unsigned int getNumVolumes() const ;
        GNode* getNode(unsigned index) const ; 
        GVolume* getVolume(unsigned int index) const ;  
        GVolume* getVolumeSimple(unsigned int index) const ;  
        const char* getPVName(unsigned int index) const ;
        const char* getLVName(unsigned int index) const ;
    public:
        void add(GMaterial* material);
        void addRaw(GMaterial* material);

     
    public:
        // via meshlib
        GMeshLib*          getMeshLib();  // unplaced meshes
        unsigned           getNumMeshes() const ;
        GItemIndex*        getMeshIndex(); 
        const GMesh*       getMesh(unsigned index) const ;  
        void               add(const GMesh* mesh);
        void countMeshUsage(unsigned meshIndex, unsigned nodeIndex);
        void reportMeshUsage(const char* msg="GGeo::reportMeshUsage");
    public:
   public:
        void traverse(const char* msg="GGeo::traverse");
    private:
        void traverse(GNode* node, unsigned int depth);
    public:
        unsigned getNumMaterials() const ;
        unsigned getNumRawMaterials() const ;
    public:
        GScene*            getScene()  ;


        NLookup*           getLookup(); 
    public:
        void  setLookup(NLookup* lookup);
    public:
        GColorizer*        getColorizer();
        OpticksColors*     getColors();
        OpticksFlags*      getFlags(); 
        OpticksResource*   getResource();
        OpticksAttrSeq*    getFlagNames(); 
        Opticks*           getOpticks() const ;
    public:
        GMaterial* getMaterial(unsigned int index) const ;   

    public:
        // m_surfacelib
        void add(GSkinSurface*  surface);
        void add(GBorderSurface*  surface);
        void addRaw(GSkinSurface* surface);
        void addRaw(GBorderSurface*  surface);
        unsigned getNumSkinSurfaces() const ;
        unsigned getNumBorderSurfaces() const ;
        unsigned getNumRawSkinSurfaces() const ;
        unsigned getNumRawBorderSurfaces() const ;
        GSkinSurface*   getSkinSurface(unsigned index) const ;  
        GBorderSurface* getBorderSurface(unsigned index) const ;  
        GSkinSurface*   findSkinSurface(const char* lv) const ;  
        GBorderSurface* findBorderSurface(const char* pv1, const char* pv2) const ;  
        void dumpSkinSurface(const char* msg="GGeo::dumpSkinSurface") const ;
        void dumpRawSkinSurface(const char* name=NULL) const ;
        void dumpRawBorderSurface(const char* name=NULL) const ;

    public:
        void findScintillatorMaterials(const char* props);
        void dumpScintillatorMaterials(const char* msg="GGeo::dumpScintillatorMaterials");
        unsigned int getNumScintillatorMaterials();
        GMaterial* getScintillatorMaterial(unsigned int index);
    public:
    public:
        GPropertyMap<float>* findRawMaterial(const char* shortname) const ;
        GProperty<float>*    findRawMaterialProperty(const char* shortname, const char* propname) const ;
        void dumpRawMaterialProperties(const char* msg="GGeo::dumpRawMaterialProperties") const ;
        std::vector<GMaterial*> getRawMaterialsWithProperties(const char* props, char delim) const ;
    public:
        gfloat3* getLow();
        gfloat3* getHigh();
        void setLow(const gfloat3& low);
        void setHigh(const gfloat3& high);
        void updateBounds(GNode* node); 
    private:
        void saveCacheMeta() const ;
        void loadCacheMeta();
    public:
        // TODO: contrast with this earlier way 
        void findCathodeMaterials(const char* props);
        void dumpCathodeMaterials(const char* msg="GGeo::dumpCathodeMaterials");
        unsigned int getNumCathodeMaterials();
        GMaterial* getCathodeMaterial(unsigned int index);
    public:
        // m_materiallib
        void setCathode(GMaterial* cathode);
        GMaterial* getCathode() const ;  
        const char* getCathodeMaterialName() const ;
    public:
        void addLVSD(const char* lv, const char* sd);
        unsigned getNumLVSD() const ;
        std::pair<std::string,std::string> getLVSD(unsigned idx) const ;
    public:
        void dumpCathodeLV(const char* msg="GGeo::dumpCathodeLV") const ;
        const char* getCathodeLV(unsigned int index) const ; 
        void getCathodeLV( std::vector<std::string>& lvnames ) const ;
        unsigned int getNumCathodeLV() const ;
        int findCathodeLVIndex(const char* lv) const ; // -1 if not found 
    public:

#if 0
    TODO: see if this can be reinstated
    public:
        void materialConsistencyCheck();
        unsigned int materialConsistencyCheck(GVolume* volume);
#endif

    public:
        void Summary(const char* msg="GGeo::Summary");
        void Details(const char* msg="GGeo::Details");

    public:
        GInstancer* getTreeCheck();
    public:
        void setPickFace(std::string pickface);
        void setPickFace(const glm::ivec4& pickface);
        void setFaceTarget(unsigned int face_index, unsigned int volume_index, unsigned int mesh_index);
        void setFaceRangeTarget(unsigned int face_index0, unsigned int face_index1, unsigned int volume_index, unsigned int mesh_index);
        glm::ivec4& getPickFace(); 
    private:
        static GGeo*                  fInstance ; 
        SLog*                         m_log ; 
        Opticks*                      m_ok ;  
        bool                          m_live ;   
        bool                          m_analytic ; 
        int                           m_gltf ; 
        Composition*                  m_composition ; 
        GInstancer*                   m_instancer ; 
        bool                          m_loaded ;  
        bool                          m_prepared ;  

        NMeta*                        m_loadedcachemeta ; 
        NMeta*                        m_lv2sd ; 


        std::vector<GVolume*>           m_sensitive_volumes ; 
        std::unordered_set<std::string> m_cathode_lv ; 

        std::vector<GMaterial*>       m_scintillators_raw ; 
        std::vector<GMaterial*>       m_cathodes_raw ; 

        NLookup*                      m_lookup ; 

        GMeshLib*                     m_meshlib ; 
        GGeoLib*                      m_geolib ; 

        GNodeLib*                     m_nodelib ; 

        GBndLib*                      m_bndlib ; 
        GMaterialLib*                 m_materiallib ; 
        GSurfaceLib*                  m_surfacelib ; 
        GScintillatorLib*             m_scintillatorlib ; 
        GSourceLib*                   m_sourcelib ; 
        GPmtLib*                      m_pmtlib ; 

        GColorizer*                   m_colorizer ; 

        gfloat3*                      m_low ; 
        gfloat3*                      m_high ; 

    private:

       // Index_t                            m_index ; 
        unsigned int                       m_sensitive_count ;  
        //GMaterial*                         m_cathode ; 
        const char*                        m_join_cfg ; 
        GJoinImpFunctionPtr                m_join_imp ;  
        GLoaderImpFunctionPtr              m_loader_imp ;  
        unsigned int                       m_mesh_verbosity ; 

    private:
        // glTF route 
        GScene*                            m_gscene ; 



};

#include "GGEO_TAIL.hh"


