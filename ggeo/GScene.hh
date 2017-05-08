#pragma once

#include <map>

class GGeo ; 
class GSolid ; 
class GBndLib ; 
class GMesh ; 

class NCSG ; 
class NScene ; 
struct nd ; 

template<class T> class NPY ;

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/*
GScene
========

Aiming to be analytic replacement for GTreeCheck 


*/

class GGEO_API GScene 
{
    public:
        GScene(GGeo* ggeo, NScene* scene);
    private:
        void init();
    private:
        void importMeshes(NScene* scene);
        GMesh* getMesh(unsigned mesh_idx);
        NCSG*  getCSG(unsigned mesh_idx);
    private:
        GSolid* createVolumeTree(NScene* scene);
        GSolid* createVolumeTree_r(nd* n);
        GSolid* createVolume(nd* n);
   private:
        unsigned int getNumRepeats(); 
        void createInstancedMergedMeshes(bool delta);
        NPY<float>*        makeInstanceTransformsBuffer(unsigned int ridx);
    private:
        GGeo*    m_ggeo ; 
        GBndLib* m_bndlib ; 
        NScene*  m_scene ; 
        GSolid*  m_root ; 
        std::map<unsigned, GMesh*> m_meshes ; 

};

#include "GGEO_TAIL.hh"


