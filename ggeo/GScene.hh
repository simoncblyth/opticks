#pragma once

#include <map>

class Opticks ; 

class GNodeLib ; 
class GItemList ; 
class GGeo ; 
class GSolid ; 
class GNode ; 
class GBndLib ; 
class GGeoLib ; 
class GMesh ; 
class GMergedMesh ; 

class NCSG ; 
class NScene ; 
struct nd ; 
struct guint4 ; 

template<class T> class NPY ;

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

class GGEO_API GScene 
{
    public:
        // ggeo currently used only for bndlib access
        GScene(Opticks* ok, GGeo* ggeo);
        GGeoLib* getGeoLib();
    private:
        void init();
    private:
        void compareTrees();
        void modifyGeometry();
        void importMeshes(NScene* scene);
        void dumpMeshes();
        GMesh* getMesh(unsigned mesh_idx);
        unsigned getNumMeshes();
        NCSG*  getCSG(unsigned mesh_idx);

        // from triangulated branch mm0
        guint4 getNodeInfo(unsigned idx);
        guint4 getIdentity(unsigned idx);
    private:
        GSolid* createVolumeTree(NScene* scene);
        GSolid* createVolumeTree_r(nd* n, GSolid* parent);
        GSolid* createVolume(nd* n);
        void addNode(GSolid* node, nd* n);
    private:
        // compare tree calculated and persisted transforms
        void           deltacheck_r( GNode* node, unsigned int depth );
    private:
        // these two methods formerly used m_ggeo to get to the m_ggeo/m_geolib 
        // now moved to holding a separate m_geolib in here
        void         checkMergedMeshes();
        void         makeMergedMeshAndInstancedBuffers() ; 

    private:
        void         makeInstancedBuffers(GMergedMesh* mergedmesh, unsigned ridx);

        NPY<float>* makeInstanceTransformsBuffer(const std::vector<GNode*>& instances, unsigned ridx);
        NPY<unsigned>* makeInstanceIdentityBuffer(const std::vector<GNode*>& instances, unsigned ridx);
        NPY<unsigned>* makeAnalyticInstanceIdentityBuffer(const std::vector<GNode*>& instances, unsigned ridx);
    private:
        GSolid*       getNode(unsigned node_idx);
    private:
        Opticks* m_ok ; 
        int      m_gltf ; 
        NScene*  m_scene ; 
        unsigned m_targetnode ; 

        GGeoLib*  m_geolib ; 
        GNodeLib* m_nodelib ; 

        GGeoLib*      m_tri_geolib ; 
        GMergedMesh*  m_tri_mm0 ; 
        GNodeLib*     m_tri_nodelib ; 
        GBndLib*      m_tri_bndlib ; 

        unsigned m_verbosity ; 

        GSolid*  m_root ; 

        std::map<unsigned, GMesh*>  m_meshes ; 
        std::map<unsigned, GSolid*> m_nodes ;  

};

#include "GGEO_TAIL.hh"


