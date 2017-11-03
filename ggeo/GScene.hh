#pragma once

#include <string>
#include <map>

class Opticks ; 
class OpticksQuery ; 
class OpticksEvent ; 

class GNodeLib ; 
class GMeshLib ; 
class GItemList ; 
class GGeo ; 
class GSolid ; 
class GNode ; 
class GBndLib ; 
class GPmtLib ; 
class GGeoLib ; 
class GMesh ; 
class GMergedMesh ; 
class GItemIndex ; 
class GColorizer ; 

class NCSG ; 
class NSensorList ; 
class NScene ; 
struct NSceneConfig ; 
struct nd ; 
struct guint4 ; 

template<class T> class NPY ;


#include "NSceneConfig.hpp"  // for enum
#include "GGeoBase.hh"

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/*
GScene
========

Canonical m_gscene instance, resident in m_ggeo,
is instanciated by GGeo::loadFromGLTF.
GMergedMesh are currently created via GGeo and
managed in its GGeoLib.

GScene.hh only used from GGeo, the actions of GScene
creating the analytic GMergedMesh are felt via 
the normal GGeoLib route. oxrap/OScene/OGeo 
(especially OGeo::makeAnalyticGeometry)
which converts the GGeo accessed GMergedMesh into OptiX form. 



Fully analytic glTF based replacement for the 
mainly triangulated GTreeCheck.

Note the only public method is the ctor, this 
gets invoked from GGeo::loadGeometry/GGeo::loadFromGLTF
when the "--gltf 1" commandline option is used with a 
value greater than 0.

Using "--gltf 4", signals an early exit following GScene 
instanciation in GGeo::loadFromGLTF.


* generally follows the same pattern as GTreeCheck 
  labelling the GNode tree with a ridx repeat index... 
  which us used within  GGeo/GGeoLib/GMergedMesh mesh merging
  (which also merges analytic solids)

* a global mm0 is needed for setting domains... when 
  operating purely instanced need to construct some 
  placeholder bbox so satisfy the global mesh 0 that 
  lots of things require

*/



class GGEO_API GScene : public GGeoBase
{
   //    friend class GGeo ;  
    public:
        static GScene* Load(Opticks* ok, GGeo* ggeo) ;
        GScene(Opticks* ok, GGeo* ggeo, bool loaded);
        //GGeoLib*  getGeoLib();
        GNodeLib* getNodeLib();

        GMergedMesh* getMergedMesh(unsigned ridx);
        GSolid* getSolid(unsigned nidx);
        void dumpNode(unsigned nidx);
        void anaEvent(OpticksEvent* evt);
        void save() const ; 

    public:
        // GGeoBase interface
        const char*       getIdentifier();
        GGeoLib*          getGeoLib() ; 
        GBndLib*          getBndLib() ; 
        GPmtLib*          getPmtLib() ; 
        GScintillatorLib* getScintillatorLib() ; 
        GSourceLib*       getSourceLib() ; 
    private:
        void initFromGLTF();
        void prepareVertexColors();
    private:
        void dumpTriInfo() const ; 
        void compareTrees() const ;
        void modifyGeometry();
        void importMeshes(NScene* scene);
        void dumpMeshes();
        void compareMeshes();
        void compareMeshes_GMeshBB();
        GMesh* getMesh(unsigned mesh_idx);
        unsigned getNumMeshes();

        NCSG* getCSG(unsigned mesh_idx);
        NCSG* findCSG(const char* soname, bool startswith) const ;
        nbbox getBBox(const char* soname, NSceneConfigBBoxType bbty) const ; 


        unsigned findTriMeshIndex(const char* soname) const ;

        // from triangulated branch mm0
        guint4 getNodeInfo(unsigned idx) const ;
        guint4 getIdentity(unsigned idx) const ;
    private:
        GSolid* createVolumeTree(NScene* scene);
        GSolid* createVolumeTree_r(nd* n, GSolid* parent, unsigned depth, bool recursive_select );
        GSolid* createVolume(nd* n, unsigned depth, bool& recursive_select );
        void transferIdentity( GSolid* node, const nd* n);
        void transferMetadata( GSolid* node, const NCSG* csg, const nd* n, unsigned depth, bool& recursive_select );
        std::string lookupBoundarySpec( const GSolid* node, const nd* n) const ;
        void addNode(GSolid* node, nd* n);
    private:
        // compare tree calculated and persisted transforms
        void           deltacheck_r( GNode* node, unsigned int depth );
    private:
        void         checkMergedMeshes();
        void         makeMergedMeshAndInstancedBuffers() ; 

    private:
        GSolid*       getNode(unsigned node_idx);
    private:
        Opticks*      m_ok ; 
        OpticksQuery* m_query ; 
        
        GGeo*    m_ggeo ; 

        NSensorList*  m_sensor_list ; 

        GGeoLib*      m_tri_geolib ; 
        GMergedMesh*  m_tri_mm0 ; 
        GNodeLib*     m_tri_nodelib ; 
        GBndLib*      m_tri_bndlib ; 
        GMeshLib*     m_tri_meshlib ; 
        GItemIndex*   m_tri_meshindex ; 

     
        bool     m_analytic ; 
        bool     m_testgeo ; 
        bool     m_loaded ; 
        bool     m_honour_selection ;

        int      m_gltf ; 
        NSceneConfig*  m_scene_config ; 
        NScene*        m_scene ; 
        int            m_num_nd ; 
        unsigned       m_targetnode ; 

        GGeoLib*      m_geolib ; 
        GNodeLib*     m_nodelib ; 
        GMeshLib*     m_meshlib ; 

        GColorizer*   m_colorizer ; 

        unsigned     m_verbosity ; 
        GSolid*      m_root ; 
        unsigned     m_selected_count ; 

        std::map<unsigned, GMesh*>   m_meshes ; 
        std::map<unsigned, GSolid*>  m_nodes ;  
        std::map<unsigned, unsigned> m_rel2abs_mesh ; 
        std::map<unsigned, unsigned> m_abs2rel_mesh ; 

};




#include "GGEO_TAIL.hh"


