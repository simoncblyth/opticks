#pragma once

#include <string>
#include <vector>

class SLog ; 
struct NSceneConfig ; 

//class GGeo ; 
class GGeoLib ;    // merged meshes 
class GNodeLib ;   // GVolume nodes
class GNode ; 
class GVolume ; 
class GBuffer ;
class GMergedMesh ;

template<class T> class Counts ;
template<class T> class NPY ;



/**
GInstancer
=============

Formerly was misnamed GTreeCheck.
Invoked by GGeo::prepareMeshes : finds instanced geometry and creates GMergedMesh for 
each instance and for the global non-instanced geometry.

Canonical instance *m_treecheck* is constituent of GGeo that is used precache 
to *createInstancedMergedMeshes* when invoked by  GGeo::loadFromG4DAE GGeo::prepareMeshes. 
This populates GGeo.m_geolib with GMergedMesh.

**/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GInstancer {
   public:
        GInstancer(GGeoLib* geolib, GNodeLib* nodelib, NSceneConfig* config) ;
        void setRepeatMin(unsigned repeat_min);
        void setVertexMin(unsigned vertex_min);
   public:
        // principal method, almost everything else invoked by this 
        void createInstancedMergedMeshes(bool deltacheck, unsigned verbosity); 
   private:
        // compare tree calculated and persisted transforms
        void           deltacheck(); 
        void           deltacheck_r( GNode* node, unsigned int depth );

   private:
        // Collecting m_repeat_candidates vector of digests
        //
        // Spin over tree counting up progenyDigests to find repeated geometry into m_digest_count
        // sort by instance counts to find the most common progenyDigests.
        // For each digest, qualify repeaters by progeny and vertex counts collecting 
        // into m_repeat_candidates, erase repeats that are contained within other repeats.
        // 
        void           traverse();
        void           traverse_r( GNode* node, unsigned int depth ); 
        void           findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min);
        bool           isContainedRepeat( const std::string& pdig, unsigned int levels ) const ;
        void           dumpRepeatCandidates(unsigned dmax);
        void           dumpRepeatCandidate(unsigned int index, bool verbose=false);
   public:
        bool           operator()(const std::string& dig) ;
   public:
        // Using m_repeat_candidates vector of digests
        //
        unsigned            getRepeatIndex(const std::string& pdig );
        unsigned            getNumRepeats();   
        GNode*              getRepeatExample(unsigned int ridx);    // first node that matches the ridx progeny digest
        std::vector<GNode*> getPlacements(unsigned int ridx);  // all GNode with the ridx progeny digest
   private:
        // recursive setRepeatIndex on the GNode tree for each of the repeated bits of geometry
        void           labelTree();
        void           labelTree_r( GNode* node, unsigned int ridx );  // recursive labelling starting from the placements
   private:
        // output side, operates via GGeo::makeMergedMesh, GGeoLib::makeMergedMesh, GMergedMesh::create
        //   GMergedMesh::traverse uses the repeat index ridx labels written into the node tree
        void           makeMergedMeshAndInstancedBuffers(unsigned verbosity);
   private:
       SLog*                     m_log ; 
       GGeoLib*                  m_geolib ; 
       unsigned                  m_verbosity ; 
       GNodeLib*                 m_nodelib ; 
       NSceneConfig*             m_config ; 

       unsigned int              m_repeat_min ; 
       unsigned int              m_vertex_min ; 
       GVolume*                   m_root ; 
       unsigned int              m_count ;  
       unsigned int              m_labels ;   // count of nodes labelled
       Counts<unsigned int>*     m_digest_count ; 
       std::vector<std::string>  m_repeat_candidates ; 
 
};


#include "GGEO_TAIL.hh"


