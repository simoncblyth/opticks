#pragma once

//#include <map>
#include <string>
#include <vector>

class GGeo ; 
class GGeoLib ; 
class GNode ; 
class GSolid ; 
class GBuffer ;
class GMergedMesh ;

template<class T> class Counts ;
template<class T> class NPY ;


// *createInstancedMergedMeshes* is canonically invoked by GGeo::loadFromG4DAE GGeo::prepareMeshes

/**
GTreeCheck
=============

TODO: rename, this does essential instancing prep, it is not a "check"

Canonical instance *m_treecheck* is member of GGeo that 
is used only precache.




**/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GTreeCheck {
   public:
       // typedef std::map<std::string, std::vector<GNode*> > MSVN ; 
   public:
        GTreeCheck(GGeo* ggeo);
        void setRepeatMin(unsigned int repeat_min);
        void setVertexMin(unsigned int vertex_min);
        void createInstancedMergedMeshes(bool deltacheck=false); 
   public:
        void init();
        void traverse();
        void deltacheck();
        unsigned int getRepeatIndex(const std::string& pdig );
        void labelTree();
        unsigned int getNumRepeats(); 
        GNode* getRepeatExample(unsigned int ridx);
   private:
        // canonically invoked by GTreeCheck::createInstancedMergedMeshes
        void makeInstancedBuffers(GMergedMesh* mergedmesh, unsigned int ridx);
        NPY<float>*        makeInstanceTransformsBuffer(unsigned int ridx);
        NPY<unsigned int>* makeInstanceIdentityBuffer(unsigned int ridx);
        NPY<unsigned int>* makeAnalyticInstanceIdentityBuffer(unsigned int ridx);
   public:
        bool operator()(const std::string& dig) ;
   private:
        void traverse( GNode* node, unsigned int depth );
        void deltacheck( GNode* node, unsigned int depth );
        void findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min);
        void dumpRepeatCandidates();
        void dumpRepeatCandidate(unsigned int index, bool verbose=false);
        bool isContainedRepeat( const std::string& pdig, unsigned int levels ) const ;
        void labelTree( GNode* node, unsigned int ridx );
        std::vector<GNode*> getPlacements(unsigned int ridx);

   private:
       GGeo*                     m_ggeo ; 
       GGeoLib*                  m_geolib ; 
       unsigned int              m_repeat_min ; 
       unsigned int              m_vertex_min ; 
       GSolid*                   m_root ; 
       unsigned int              m_count ;  
       unsigned int              m_labels ;  
       Counts<unsigned int>*     m_digest_count ; 
       std::vector<std::string>  m_repeat_candidates ; 
 
};


#include "GGEO_TAIL.hh"



