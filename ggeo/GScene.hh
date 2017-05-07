#pragma once

class GGeo ; 
class NScene ; 
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
       unsigned int getNumRepeats(); 
   private:
       void createInstancedMergedMeshes(bool delta);
       //void checkInstancedBuffers(GMergedMesh* mergedmesh, unsigned int ridx);
   private:
       //void makeInstancedBuffers(GMergedMesh* mergedmesh, unsigned int ridx);
       NPY<float>*        makeInstanceTransformsBuffer(unsigned int ridx);
       //NPY<unsigned int>* makeInstanceIdentityBuffer(unsigned int ridx);
       //NPY<unsigned int>* makeAnalyticInstanceIdentityBuffer(unsigned int ridx);
    private:
       GGeo*   m_ggeo ; 
       NScene* m_scene ; 
};

#include "GGEO_TAIL.hh"


