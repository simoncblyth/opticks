/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <unordered_set>
#include <iterator>

#include <glm/fwd.hpp>
#include "plog/Severity.h"

// npy-
template <typename T> class NPY ;
#include "NConfigurable.hpp"

class NLookup ; 
class BMeta ;
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
class GParts ; 

class GInstancer ;
class GColorizer ; 

class GItemIndex ; 
class GItemList ; 
class GMergedMesh ;

struct GGeoDump ; 


#include "SGeo.hh"
#include "GGeoBase.hh"

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**
GGeo
=====

**/

class GGEO_API GGeo : public GGeoBase, public NConfigurable, public SGeo {
    public:
        friend class  X4PhysicalVolume ;  // X4PhysicalVolume::init needs afterConvertMaterial 
        friend class  AssimpGGeo ; 
        friend struct GSceneTest ; 
    public:
        static const plog::Severity LEVEL ; 
        static GGeo* Get();  // statically provides the last instanciated GGeo instance
        static GGeo* Load(Opticks* ok); 
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
        GMeshLib*         getMeshLib() const ;

        GBndLib*          getBndLib() const ; 
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
        void   setGDMLAuxMeta(const BMeta* gdmlauxmeta); 
        const BMeta* getGDMLAuxMeta() const ; 
    public:
        Composition* getComposition();
        void setComposition(Composition* composition);
    public:
        bool isLoadedFromCache() const ;
        void loadGeometry(); 
        void loadFromCache();
        void postDirectTranslation();  // from G4Opticks::translateGeometry
    private: 
        void postLoadFromCache();
        void postDirectTranslationDump() const ; 
    private: 
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
    private:
        void prepareOpticks(); 
        void deferred(); 
        void deferredCreateGParts(); 
        void deferredCreateGGeoDump();
    public:
        GParts* getCompositeParts(unsigned index) const ;
        void dumpParts(const char* msg, int repeatIdx, int primIdx, int partIdxRel) const ;
    public:
        // via m_bndlib
        unsigned int getMaterialLine(const char* shortname);
        std::string  getSensorBoundaryReport() const ; 

        std::string getBoundaryName(unsigned boundary) const ; // 0-based boundary index argument
        bool     isSameMaterialBoundary(unsigned boundary) const ; 
        unsigned getBoundary(const char* spec) const ; // 0-based, 0xffffffff UNSET
        int      getSignedBoundary(const char* spec) const ; // 1-based, 0 UNSET    
        int      getSignedBoundary() const ; 
   private:
        void init(); 
        void initLibs(); 
    public:
        void save();
        void anaEvent(OpticksEvent* evt);
    private:
    public:
        // pass thru to geolib
        GMergedMesh* makeMergedMesh(unsigned int index, const GNode* base, const GNode* root);
        unsigned int getNumMergedMesh() const ;
    public:
        // these are operational from cache
        void dumpTree(const char* msg="GGeo::dumpTree");  
        void dumpVolume(unsigned int index, const char* msg="GGeo::dumpVolume");  
        void dumpNodeInfo(unsigned int mmindex, const char* msg="GGeo::dumpNodeInfo" );
        std::string dumpStats(const char* msg="GGeo::dumpStats");

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

   public:
        // via GNodeLib
        void           setRootVolume(const GVolume* volume); 
        const GVolume* getRootVolume() const ; 
   public:
        // via GNodeLib
        void           addVolume(const GVolume* volume);
   public:
        unsigned int   getNumVolumes() const ;
        const GNode*   getNode(unsigned index) const ; 
        NPY<float>*    getTransforms() const ;
        NPY<float>*    getInverseTransforms() const ;
        const GVolume* getVolume(unsigned int index) const ;  
        const GVolume* getVolumeSimple(unsigned int index) const ;  
        const char*    getPVName(unsigned int index) const ;
        const char*    getLVName(unsigned int index) const ;
    public:
        // from GGeoLib
        glm::uvec4 getIdentity( unsigned ridx, unsigned pidx, unsigned oidx, bool check=true) const ;
        glm::mat4  getTransform(unsigned ridx, unsigned pidx, unsigned oidx, bool check=true) const ;  
        unsigned   getNumRepeats() const ; 
        unsigned   getNumPlacements(unsigned ridx) const ;
        unsigned   getNumVolumes(unsigned ridx) const ;
        void       dumpShape(const char* msg="GGeo::dumpShape") const ; 

    public:
        // via GNodeLib
        unsigned getNumTransforms() const ; 
        glm::uvec4 getIdentity( unsigned nidx) const ;
        glm::uvec4 getNRPO(unsigned nidx) const ;
        glm::mat4 getTransform(unsigned nidx) const ;  
        glm::mat4 getInverseTransform(unsigned nidx) const ;  
        void dumpVolumes(const std::map<std::string, int>& targets, const char* msg="GGeo::dumpVolumes", float extent_cut_mm=5000., int cursor=-1 ) const ;
        glm::vec4 getCE(unsigned index) const ; 
        int findNodeIndex(const void* origin) const ; 
    public:
        void dumpNode(unsigned nidx); 
        void dumpNode(unsigned ridx, unsigned pidx, unsigned oidx); 
    public:
        // sensor handling via GNodeLib
        unsigned       getNumSensorVolumes() const ;                    // pre-cache and post-cache
        glm::uvec4     getSensorIdentity(unsigned sensorIndex) const ;  // pre-cache and post-cache
        unsigned       getSensorIdentityStandin(unsigned sensorIndex) const ;  // pre-cache and post-cache

        const GVolume* getSensorVolume(unsigned sensorIndex) const ;    // pre-cache only 
        std::string    reportSensorVolumes(const char* msg) const ; 
        void           dumpSensorVolumes(const char* msg) const ; 
        void           getSensorPlacements(std::vector<void*>& placements, bool outer_volume) const ;
    public:
        // node indices via GNodeLib 
        int            getFirstNodeIndexForGDMLAuxTargetLVName() const ;
        void           getNodeIndicesForLVName(std::vector<unsigned>& nidxs, const char* lvname) const ;
        void           dumpNodes(const std::vector<unsigned>& nidxs, const char* msg="GGeo::dumpNodes") const ; 
        int            getFirstNodeIndexForPVName(const char* pvname) const ;
        int            getFirstNodeIndexForPVNameStarting(const char* pvname_start) const ;
        int            getFirstNodeIndexForPVName() const ;  // --pvname 
        int            getFirstNodeIndexForTargetPVName() const ;    // --targetpvn 
    public:
        void add(GMaterial* material);
        void addRaw(GMaterial* material);

     
    public:
        // via meshlib
        GMeshLib*          getMeshLib();  // unplaced meshes
        unsigned           getNumMeshes() const ;
        const char*        getMeshName(unsigned midx) const ;
        void               getMeshNames(std::vector<std::string>& meshNames) const ;
        int                getMeshIndexWithName(const char* name, bool startswith=true) const ;
        std::string        descMesh() const ;


#ifdef OLD_INDEX
        GItemIndex*        getMeshIndex(); 
#endif
        const GMesh*       getMesh(unsigned index) const ;  
        void               add(const GMesh* mesh);
        void countMeshUsage(unsigned meshIndex, unsigned nodeIndex);
        void reportMeshUsage(const char* msg="GGeo::reportMeshUsage") const ;
    public:
   public:
        void traverse(const char* msg="GGeo::traverse");
    private:
        void traverse(const GNode* node, unsigned depth);
    public:
        unsigned getNumMaterials() const ;
        unsigned getNumRawMaterials() const ;
    public:

#ifdef OLD_SCENE
        GScene*            getScene()  ;
#endif

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
        GPropertyMap<float>* findMaterial(const char* shortname) const ;
        GPropertyMap<float>* findRawMaterial(const char* shortname) const ;
        GProperty<float>*    findRawMaterialProperty(const char* shortname, const char* propname) const ;
        void dumpRawMaterialProperties(const char* msg="GGeo::dumpRawMaterialProperties") const ;
        std::vector<GMaterial*> getRawMaterialsWithProperties(const char* props, char delim) const ;
    public:

#ifdef OLD_BOUNDS
        gfloat3* getLow();
        gfloat3* getHigh();
        void setLow(const gfloat3& low);
        void setHigh(const gfloat3& high);
        void updateBounds(GNode* node); 
#endif
    private:
        void saveCacheMeta() const ;
        void loadCacheMeta();
        void saveGLTF() const ;
    public:
        // TODO: contrast with this ancient earlier way 
        void findCathodeMaterials(const char* props);
        void dumpCathodeMaterials(const char* msg="GGeo::dumpCathodeMaterials");
        unsigned int getNumCathodeMaterials();
        GMaterial* getCathodeMaterial(unsigned int index);

    public:
        void Summary(const char* msg="GGeo::Summary");
        void Details(const char* msg="GGeo::Details");

    public:
        GInstancer* getInstancer() const ;
    public:
        void dryrun_convert() ;
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
        bool                          m_enabled_legacy_g4dae ; 
        bool                          m_live ;   
        Composition*                  m_composition ; 
        GInstancer*                   m_instancer ; 
        bool                          m_loaded_from_cache ;  
        bool                          m_prepared ;  

        const BMeta*                  m_gdmlauxmeta ;  
        BMeta*                        m_loadedcachemeta ; 
        BMeta*                        m_lv2sd ; 
        BMeta*                        m_lv2mt ; 
        const char*                   m_origin_gdmlpath ; 

        std::vector<GVolume*>           m_sensor_volumes ; 
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

    private:

        unsigned int                       m_sensitive_count ;  
        const char*                        m_join_cfg ; 
        GJoinImpFunctionPtr                m_join_imp ;  
        GLoaderImpFunctionPtr              m_loader_imp ;  
        unsigned int                       m_mesh_verbosity ; 

    private:
        bool                               m_save_mismatch_placements ;
        GGeoDump*                          m_dump ; 
        int                                m_placeholder_last ; 

};

#include "GGEO_TAIL.hh"


