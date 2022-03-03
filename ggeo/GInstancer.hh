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

#include <string>
#include <vector>
#include <map>
#include <set>
#include "plog/Severity.h"

class SLog ; 
class Opticks ;

class GGeo ; 
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

Canonical instance *m_instancer* is constituent of GGeo that is used precache 
to *createInstancedMergedMeshes* when invoked by GGeo::prepareVolumes 
This populates GGeo.m_geolib with GMergedMesh.

**/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GInstancer {
   public:
        static const plog::Severity LEVEL ;  
        static const int INSTANCE_REPEAT_MIN ; // GInstancer_instance_repeat_min
        static const int INSTANCE_VERTEX_MIN ; // GInstancer_instance_vertex_min 
   public:
        GInstancer(Opticks* ok, GGeo* ggeo ) ;
        void setRepeatMin(unsigned repeat_min);
        void setVertexMin(unsigned vertex_min);
   public:
        // principal method, almost everything else invoked by this 
        void createInstancedMergedMeshes(bool deltacheck, unsigned verbosity); 
   private:
        void           initRoot();  
        // compare tree calculated and persisted transforms
        void           deltacheck(); 
        void           deltacheck_r( const GNode* node, unsigned int depth );

   private:
        // Collecting m_repeat_candidates vector of digests
        //
        // Spin over tree counting up progenyDigests to find repeated geometry into m_digest_count
        // sort by instance counts to find the most common progenyDigests.
        // For each digest, qualify repeaters by progeny and vertex counts collecting 
        // into m_repeat_candidates, erase repeats that are contained within other repeats.
        // 
        void           traverse();
        void           traverse_r( const GNode* node, unsigned int depth ); 
        void           findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min);
        void           sortRepeatCandidates();
        bool           isContainedRepeat( const std::string& pdig, unsigned int levels ) const ;
        void           dumpRepeatCandidates(unsigned dmax);
        void           dumpRepeatCandidate(unsigned int index, bool verbose=false);

        void           dumpDigests(const std::vector<std::string>& digs, const char* msg="GInstancer::dumpDigests") ;


   private: 
        // populate GNodeLib 
        void           collectNodes();
        void           collectNodes_r( const GNode* node, unsigned depth );
   public:
        bool           operator()(const std::string& dig) ;
   public:
        // Using m_repeat_candidates vector of digests
        //
        unsigned            getRepeatIndex(const std::string& pdig );
        unsigned            getNumRepeats() const ;   
        const GNode*        getRepeatExample(unsigned ridx);     // first node that matches the ridx progeny digest
        const GNode*        getLastRepeatExample(unsigned ridx); // last node that matches the ridx progeny digest
        std::vector<const GNode*> getPlacements(unsigned int ridx);  // all GNode with the ridx progeny digest
   public:
        void dump(const char* msg) const ;
   private:
        void dumpMeshset() const ;
        void dumpCSGSkips() const ;
   private:
        // recursive setRepeatIndex on the GNode tree for each of the repeated bits of geometry
        void           labelTree();
        void           labelRepeats_r( GNode* node, unsigned ridx, unsigned pidx, int outernode_copyNo, const GVolume* outer_volume  );  // recursive labelling starting from the placements
        void           labelGlobals_r( GNode* node, unsigned depth );  // recursive labelling starting from root of only ridx 0 nodes
        void           visitNode( GNode* node, unsigned ridx ) ; 
   private:
        // output side, operates via GGeo::makeMergedMesh, GGeoLib::makeMergedMesh, GMergedMesh::create
        //   GMergedMesh::traverse uses the repeat index ridx labels written into the node tree
        void           makeMergedMeshAndInstancedBuffers(unsigned verbosity);
   private:
       SLog*                     m_log ; 
       Opticks*                  m_ok ; 
       unsigned                  m_repeat_min ; 
       unsigned                  m_vertex_min ; 

       GGeo*                     m_ggeo ; 
       GGeoLib*                  m_geolib ; 
       GNodeLib*                 m_nodelib ; 

       unsigned                  m_verbosity ; 

       const GVolume*            m_root ; 
       GVolume*                  m_root_ ; 
       unsigned int              m_count ;  
       unsigned int              m_labels ;   // count of nodes labelled
       Counts<unsigned int>*     m_digest_count ; 
       std::vector<std::string>  m_repeat_candidates ; 
   

       typedef std::set<unsigned> SU ; 
       typedef std::vector<unsigned> VU ; 
       typedef std::map<unsigned, SU> MUSU ; 
       typedef std::map<unsigned, VU> MUVU ; 

       MUSU        m_meshset ;   // collect unique mesh indices (LVs) for each ridx     
       MUVU        m_csgskiplv ;   // collect node indices for each skipped LVIdx     
       unsigned    m_csgskiplv_count ;

       unsigned    m_repeats_count ; 
       unsigned    m_globals_count ; 
       unsigned    m_offset_count ;  // within the instance  
       bool        m_duplicate_outernode_copynumber ; 
 
};


#include "GGEO_TAIL.hh"


