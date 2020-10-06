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


#include "SLog.hh"
#include "BTimeKeeper.hh"

// npy-
#include "NSceneConfig.hpp"
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSensor.hpp"
#include "Counts.hpp"

#include "Opticks.hh"

#include "GTreePresent.hh"
#include "GMergedMesh.hh"


#include "GGeoLib.hh"
#include "GNodeLib.hh"

#include "GVolume.hh"
#include "GMatrix.hh"
#include "GBuffer.hh"
#include "GTree.hh"
#include "GInstancer.hh"
#include "GPt.hh"


#include "PLOG.hh"



const plog::Severity GInstancer::LEVEL = PLOG::EnvLevel("GInstancer", "DEBUG") ; 

GInstancer::GInstancer(Opticks* ok, GGeoLib* geolib, GNodeLib* nodelib, NSceneConfig* config) 
    : 
    m_log(new SLog("GInstancer::GInstancer","", verbose)),
    m_ok(ok),
    m_global_instance_enabled(m_ok->isGlobalInstanceEnabled()), // --global_instance_enabled
    m_geolib(geolib),
    m_verbosity(geolib->getVerbosity()),
    m_nodelib(nodelib),
    m_config(config),
    m_repeat_min(config->instance_repeat_min),
    m_vertex_min(config->instance_vertex_min),  // aiming to include leaf? sStrut and sFasteners
    m_root(NULL),
    m_count(0),
    m_labels(0),
    m_digest_count(new Counts<unsigned>("progenyDigest")),
    m_csgskiplv_count(0),
    m_repeats_count(0),
    m_globals_count(0), 
    m_duplicate_outernode_copynumber(true)
{
}

unsigned GInstancer::getNumRepeats() const 
{
    return m_repeat_candidates.size();
}


void GInstancer::setRepeatMin(unsigned repeat_min)
{
    m_repeat_min = repeat_min ; 
}

void GInstancer::setVertexMin(unsigned vertex_min)
{
    m_vertex_min = vertex_min ; 
}

/**
GInstancer::createInstancedMergedMeshes
------------------------------------------

Canonical invokation from GGeo::prepareVolumes

1. spin over tree counting up progenyDigests to find repeated geometry 
2. recursive setRepeatIndex on the GNode tree for each of the repeated bits of geometry
3. makeMergedMeshAndInstancedBuffers


**/

void GInstancer::createInstancedMergedMeshes(bool delta, unsigned verbosity)
{
    OK_PROFILE("GInstancer::createInstancedMergedMeshes"); 

    if(delta) deltacheck();

    OK_PROFILE("GInstancer::createInstancedMergedMeshes:deltacheck");

    traverse();   // spin over tree counting up progenyDigests to find repeated geometry 

    OK_PROFILE("GInstancer::createInstancedMergedMeshes:traverse");

    labelTree();  // recursive setRepeatIndex on the GNode tree for each of the repeated bits of geometry

    OK_PROFILE("GInstancer::createInstancedMergedMeshes:labelTree");

    makeMergedMeshAndInstancedBuffers(verbosity);

    OK_PROFILE("GInstancer::createInstancedMergedMeshes:makeMergedMeshAndInstancedBuffers");

    //if(t.deltaTime() > 0.1)
    //t.dump("GInstancer::createInstancedMergedMeshes deltaTime > cut ");
}



/**
GInstancer::traverse
---------------------

Find repeated subtrees by comparing progeny digests from every node of the tree, 
with contained subtree repeats being excluded.


DYB: minrep 120 removes repeats from headonPMT, calibration sources and RPC leaving just PMTs 

**/

void GInstancer::traverse()
{
    LOG(LEVEL) << "[" ; 

    m_root = m_nodelib->getVolume(0);
    m_root_ = const_cast<GVolume*>(m_root); 


    assert(m_root);

    // count occurences of distinct progeny digests (relative sub-tree identities) in m_digest_count 
    traverse_r(m_root, 0);

    m_digest_count->sort(false);   // descending count order, ie most common subtrees first
    //m_digest_count->dump();

    // collect digests of repeated pieces of geometry into  m_repeat_candidates
    findRepeatCandidates(m_repeat_min, m_vertex_min); 

    unsigned num_reps = getNumRepeats();
    if(num_reps > 0 )
    dumpRepeatCandidates(20u);

    LOG(LEVEL) << "]" ; 
}

void GInstancer::traverse_r( const GNode* node, unsigned int depth)
{
    GNode* no = const_cast<GNode*>(node);  
    std::string& pdig = no->getProgenyDigest();
    m_digest_count->add(pdig.c_str());
    m_count++ ; 

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1 );
}




/**
GInstancer::deltacheck
-----------------------

Check consistency of the level transforms

**/


void GInstancer::deltacheck()
{
    m_root = m_nodelib->getVolume(0);
    assert(m_root);

    deltacheck_r(m_root, 0);
}

void GInstancer::deltacheck_r( const GNode* node, unsigned int depth)
{
    const GVolume* volume = dynamic_cast<const GVolume*>(node) ;
    GMatrixF* gtransform = volume->getTransform();

    // volumes levelTransform is set in AssimpGGeo and hails from the below with level -2
    //      aiMatrix4x4 AssimpNode::getLevelTransform(int level)
    //  looks to correspond to the placement of the LV within its PV  

    //GMatrixF* ltransform = volume->getLevelTransform();  

    GVolume* vol = const_cast<GVolume*>(volume);    // due to progeny cache 
    GMatrixF* ctransform = vol->calculateTransform();
    float delta = gtransform->largestDiff(*ctransform);

    
    unsigned int nprogeny = node->getPriorProgenyCount() ;

    if(nprogeny > 0 ) 
        LOG(debug) 
            << " #progeny "  << std::setw(6) << nprogeny 
            << " delta*1e6 " << std::setprecision(6) << std::fixed << delta*1e6 
            << " name " << node->getName() 
            ;

    assert(delta < 1e-6) ;

    for(unsigned int i = 0; i < node->getNumChildren(); i++) deltacheck_r(node->getChild(i), depth + 1 );
}





struct GRepeat
{
    unsigned    repeat_min ; 
    unsigned    vertex_min ; 
    unsigned    index ; 
    std::string pdig ; 
    unsigned    ndig ; 
    GNode*      first ;   // cannot const as collection is deferred
    unsigned    nprog ; 
    unsigned    nvert ; 
    bool        candidate ; 
    bool        select ; 

    bool isListed(const std::vector<std::string>& pdigs_)
    {
        return std::find(pdigs_.begin(), pdigs_.end(), pdig ) != pdigs_.end() ;   
    }

    GRepeat( unsigned repeat_min_, unsigned vertex_min_, unsigned index_, const std::string& pdig_, unsigned ndig_, GNode* first_ ) 
          :
          repeat_min(repeat_min_),
          vertex_min(vertex_min_),
          index(index_),
          pdig(pdig_), 
          ndig(ndig_), 
          first(first_),
          nprog(first->getPriorProgenyCount()),
          nvert(first->getProgenyNumVertices()),
          // includes self when GNode.m_selfdigest is true
          candidate(ndig > repeat_min && nvert > vertex_min ),
          select(false)
    {
    }

    std::string desc()
    { 
        std::stringstream ss ; 
        ss    << ( candidate ? " ** " : "    " ) 
              << ( select    ? " ## " : "    " ) 
              << " idx "   << std::setw(3) << index 
              << " pdig "  << std::setw(32) << pdig  
              << " ndig "  << std::setw(6) << ndig
              << " nprog " <<  std::setw(6) << nprog 
              << " nvert " <<  std::setw(6) << nvert
              << " n "     <<  first->getName() 
              ;  
        return ss.str();
    }

};



/**
GInstancer::findRepeatCandidates
----------------------------------

Suspect problem with allowing leaf repeaters is that digesta are not-specific enough, 
so get bad matching. 

Allowing leaf repeaters results in too many, so place vertex count reqirement too. 


* m_repeat_candidates is a vector of string digests
* std::remove_if invokes the UnaryPredicate GInstancer::operator() with digest argument
  to decide if a repeat is contained within another and hence is disqualified.


**/

void GInstancer::findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min)
{
    LOG(LEVEL) << "[" ; 

    unsigned int nall = m_digest_count->size() ; 
    std::vector<GRepeat> cands ; 

    // over distinct subtrees (ie progeny digests)
    for(unsigned int i=0 ; i < nall ; i++)
    {
        std::pair<std::string,unsigned int>&  kv = m_digest_count->get(i) ;

        std::string& pdig = kv.first ; 
        unsigned int ndig = kv.second ;  // number of occurences of the progeny digest 

        GNode* first = m_root_->findProgenyDigest(pdig) ; // first node that matches the progeny digest

        GRepeat cand(repeat_min, vertex_min,  i, pdig, ndig , first );
        cands.push_back(cand) ;

        if(cand.candidate) m_repeat_candidates.push_back(pdig);        
    }

    unsigned num_all = cands.size() ; 
    assert( num_all == nall ); 

    // erase repeats that are enclosed within other repeats 
    // ie that have an ancestor which is also a repeat candidate
    m_repeat_candidates.erase(
         std::remove_if(m_repeat_candidates.begin(), m_repeat_candidates.end(), *this ), 
         m_repeat_candidates.end()
    ); 

    unsigned num_repcan = m_repeat_candidates.size() ; 

    LOG(info) 
              << " nall " << nall 
              << " repeat_min " << repeat_min 
              << " vertex_min " << vertex_min 
              << " num_repcan " << num_repcan
              ;

    if(num_repcan > 0)
    {
        unsigned dmax = 30u ;  
        LOG(info) 
                  << " num_all " << num_all 
                  << " num_repcan " << num_repcan 
                  << " dmax " << dmax
                  ;
        std::cout << " (**) candidates fulfil repeat/vert cuts   "  << std::endl ;
        std::cout << " (##) selected survive contained-repeat disqualification " << std::endl ;

        for(unsigned i=0 ; i < std::min(num_all, dmax) ; i++)
        {
            GRepeat& cand = cands[i];
            cand.select = cand.isListed(m_repeat_candidates) ;
            std::cout << cand.desc() << std::endl; 
        }
    }
    LOG(LEVEL) << "]" ; 
}




bool GInstancer::operator()(const std::string& dig)  
{
    bool cr = isContainedRepeat(dig, 3);
 
    if(cr && m_verbosity > 2) 
         LOG(info) 
                  << "GInstancer::operator() "
                  << " pdig "  << std::setw(32) << dig  
                  << " disallowd as isContainedRepeat "
                  ;

    return cr ;  
} 

/**
GInstancer::isContainedRepeat
------------------------------

For the first node that matches the *pdig* progeny digest
look back *levels* ancestors to see if any of the immediate ancestors 
are also repeat candidates, if they are then this is a contained repeat
and is thus disallowed in favor of the ancestor that contains it. 

**/

bool GInstancer::isContainedRepeat( const std::string& pdig, unsigned int levels ) const 
{
    GNode* node = m_root_->findProgenyDigest(pdig) ; 
    std::vector<GNode*>& ancestors = node->getAncestors();  // ordered from root to parent 
    unsigned int asize = ancestors.size(); 

    for(unsigned int i=0 ; i < std::min(levels, asize) ; i++)
    {
        GNode* a = ancestors[asize - 1 - i] ; // from back of ancestors to start with parent
        std::string& adig = a->getProgenyDigest();
        if(std::find(m_repeat_candidates.begin(), m_repeat_candidates.end(), adig ) != m_repeat_candidates.end())
        { 
            return true ;
        }
    }
    return false ; 
} 


void GInstancer::dumpRepeatCandidates(unsigned dmax)
{
    unsigned num_repcan = m_repeat_candidates.size() ; 
    LOG(info) 
              << " num_repcan " << num_repcan
              << " dmax " << dmax
               ;
    for(unsigned int i=0 ; i < std::min(num_repcan, dmax) ; i++) dumpRepeatCandidate(i) ;
}


void GInstancer::dumpRepeatCandidate(unsigned int index, bool verbose)
{
    std::string pdig = m_repeat_candidates[index];
    unsigned int ndig = m_digest_count->getCount(pdig.c_str());

    GNode* first = m_root_->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    std::vector<const GNode*> placements = m_root_->findAllProgenyDigest(pdig);
    std::cout  
                  << " pdig "  << std::setw(32) << pdig  
                  << " ndig "  << std::setw(6) << std::dec << ndig
                  << " nprog " <<  std::setw(6) << std::dec << first->getPriorProgenyCount() 
                  << " placements " << std::setw(6) << placements.size()
                  << " n "          <<  first->getName() 
                  << std::endl 
                  ;  

    assert(placements.size() == ndig ); // restricting traverse to just selected causes this to fail
    if(verbose)
    {
        for(unsigned int i=0 ; i < placements.size() ; i++)
        {
            const GNode* place = placements[i] ;
            GMatrix<float>* t = place->getTransform();
            std::cout 
                   << " t " << t->brief() 
                   << std::endl 
                   ;  
        }
    }
}

unsigned int GInstancer::getRepeatIndex(const std::string& pdig )
{
    // repeat index corresponding to a digest
     unsigned int index(0);
     std::vector<std::string>::iterator it = std::find(m_repeat_candidates.begin(), m_repeat_candidates.end(), pdig );
     if(it != m_repeat_candidates.end())
     {
         index = std::distance(m_repeat_candidates.begin(), it ) + 1;  // 1-based index
         LOG(debug)<<"GInstancer::getRepeatIndex " 
                  << std::setw(32) << pdig 
                  << " index " << index 
                  ;
     }
     return index ; 
}


/**
GInstancer::labelTree
-----------------------

Sets the repeat index, by recursive labelling 
starting from each of the placements.

The copyNumber from the outernodes of the instance placement 
nodes are copied to all of the nodes of the instance subtree, 
as in some sense the copyNumber is relevant to all nodes of the subtrees
of each placement.  

Notice that the copyNumber is distinct for each placement 
(and indeed is used as a "pmtID" by JUNO) as are the transforms 
for these placements. 

Due to this special handling of instance identity and transforms is required, 
done in  GMergedMesh::addInstancedBuffers.

**/

void GInstancer::labelTree()   // hmm : doesnt label global volumes ?
{
    LOG(LEVEL) << "[" 
               << " nrep " << m_repeat_candidates.size() 
               ; 

    m_labels = 0 ; 

    for(unsigned int i=0 ; i < m_repeat_candidates.size() ; i++)
    {
         std::string pdig = m_repeat_candidates[i];
         unsigned int ridx = getRepeatIndex(pdig);
         assert(ridx == i + 1 );
         std::vector<const GNode*> placements = m_root_->findAllProgenyDigest(pdig);

         LOG(LEVEL) 
               << " i " << std::setw(2) << i 
               << " ridx " << std::setw(2) << ridx 
               << " placements " << std::setw(7) << placements.size()
               ; 

         // recursive labelling starting from the placements
         for(unsigned int p=0 ; p < placements.size() ; p++)
         {
             const GNode* outernode = placements[p] ; 
             const GVolume* outer_volume = dynamic_cast<const GVolume*>(outernode) ; 
             int outernode_copyNumber = outer_volume->getCopyNumber() ; 
             GNode* start = const_cast<GNode*>(outernode) ; 
             labelRepeats_r(start, ridx, outernode_copyNumber, outer_volume );
         }
    }

    
    assert(m_root);
    traverseGlobals_r(m_root, 0);

    LOG((m_csgskiplv_count > 0 ? fatal : LEVEL))
        << " m_labels (count of non-zero setRepeatIndex) " << m_labels 
        << " m_csgskiplv_count " << m_csgskiplv_count
        << " m_repeats_count " << m_repeats_count
        << " m_globals_count " << m_globals_count
        << " total_count : " << ( m_globals_count + m_repeats_count ) 
        ;     

    LOG(LEVEL) << "]" ; 

}

/**
GInstancer::labelRepeats_r
----------------------------


**/

void GInstancer::labelRepeats_r( GNode* node, unsigned int ridx, int outernode_copyNumber, const GVolume* outer_volume )
{
    GVolume* vol = dynamic_cast<GVolume*>(node); 
    node->setRepeatIndex(ridx);
    m_repeats_count += 1 ; 

    vol->setOuterVolume(outer_volume) ; 

    if(m_duplicate_outernode_copynumber && outernode_copyNumber > -1)
    {
        if( vol->getCopyNumber() != unsigned(outernode_copyNumber) )
        {
            vol->setCopyNumber(outernode_copyNumber); 
        }    
    } 

    unsigned lvidx = node->getMeshIndex();  
    m_meshset[ridx].insert( lvidx ) ; 

    if( m_ok->isCSGSkipLV(lvidx) )   // --csgskiplv
    {
        vol->setCSGSkip(true);      

        m_csgskiplv[lvidx].push_back( node->getIndex() ); 
        m_csgskiplv_count += 1 ; 
    }

    if(ridx > 0)
    {
         LOG(debug)
             << " ridx " << std::setw(5) << ridx
             << " n " << node->getName()
             ;
         m_labels++ ; 
    }
    for(unsigned int i = 0; i < node->getNumChildren(); i++) labelRepeats_r(node->getChild(i), ridx, outernode_copyNumber, outer_volume );
}


/**
GInstancer::traverseGlobals_r
-------------------------------

Only recurses whilst in global territory with ridx == 0, as soon as hit a repeated 
volume, with ridx > 0, stop recursing. 

Skipping of an instanced LV is done here by setting a flag.

Currently trying to skip a global lv at this rather late juncture 
leads to inconsistencies manifesting in a corrupted color buffer 
(i recall that all global volumes are retained for index consistency in the merge of GMergedMesh GGeoLib)
so moved to simply editing the input GDML
Presumably could also do this by moving the skip earlier to the Geant4 X4 traverse
see notes/issues/torus_replacement_on_the_fly.rst


**/

void GInstancer::traverseGlobals_r( const GNode* node, unsigned depth )
{
    unsigned ridx = node->getRepeatIndex() ; 
    if( ridx > 0 ) return ; 
    assert( ridx == 0 ); 
    m_globals_count += 1 ; 

    unsigned lvidx = node->getMeshIndex();  
    m_meshset[ridx].insert( lvidx ) ; 

/*
    if( m_ok->isCSGSkipLV(lvidx) )   // --csgskiplv
    {
        assert(0 && "skipping of LV used globally, ie non-instanced, is not currently working "); 

        GVolume* vol = dynamic_cast<GVolume*>(node); 
        vol->setCSGSkip(true);      

        m_csgskiplv[lvidx].push_back( node->getIndex() ); 
        m_csgskiplv_count += 1 ; 
    }
*/  
 
    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverseGlobals_r(node->getChild(i), depth + 1 );
}


/**
GInstancer::getPlacements
--------------------------




**/


std::vector<const GNode*> GInstancer::getPlacements(unsigned ridx)
{
    std::vector<const GNode*> placements ;
    if(ridx == 0)
    {
        placements.push_back(m_root);
    }
    else
    {
        assert(ridx >= 1); // ridx is a 1-based index
        assert(ridx-1 < m_repeat_candidates.size()); 
        std::string pdig = m_repeat_candidates[ridx-1];
        placements = m_root_->findAllProgenyDigest(pdig);
    } 
    return placements ; 
}


const GNode* GInstancer::getRepeatExample(unsigned ridx)
{
    std::vector<const GNode*> placements = getPlacements(ridx);
    std::string pdig = m_repeat_candidates[ridx-1];
    GNode* node = m_root_->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    assert(placements[0] == node);

    return node ; 
}

const GNode* GInstancer::getLastRepeatExample(unsigned ridx)
{
    std::vector<const GNode*> placements = getPlacements(ridx);
    std::string pdig = m_repeat_candidates[ridx-1];
    GNode* node = m_root_->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    assert(placements[0] == node);

    const GVolume* first = static_cast<const GVolume*>(placements.front()) ; 
    const GVolume* last = static_cast<const GVolume*>(placements.back()) ; 

    LOG(info) 
        << " ridx " << ridx
        << std::endl 
        << " first.pt " << first->getPt()->desc() 
        << std::endl 
        << " last.pt  " << last->getPt()->desc() 
        ; 

    return placements.back() ; 
}






/**
GInstancer::makeMergedMeshAndInstancedBuffers
----------------------------------------------

Populates m_geolib with merged meshes including the instancing buffers.

Notice that for repeated geometry subtrees only the first example of such an 
instance is concatenated into a GMergedMesh, as they are all the same.  Other than 
this just the placement transforms for each instance are needed. These are 
added with GMergedMesh::addInstancedBuffers.

Using *last=true* is for the ndIdx of GParts(GPts) to match 
those of GParts(NCSG) see notes/issues/x016.rst

**/

void GInstancer::makeMergedMeshAndInstancedBuffers(unsigned verbosity)
{
    bool last = false ; 

    const GNode* root = m_nodelib->getNode(0);
    assert(root); 
    GNode* base = NULL ; 


    // passes thru to GMergedMesh::create with management of the mm in GGeoLib
    unsigned ridx0 = 0 ; 
    GMergedMesh* mm0 = m_geolib->makeMergedMesh(ridx0, base, root, verbosity, false );


    std::vector<const GNode*> placements = getPlacements(ridx0);  // just m_root
    assert(placements.size() == 1 );
    mm0->addInstancedBuffers(placements);  // call for global for common structure 

    unsigned numRepeats = getNumRepeats();
    unsigned numRidx = 1 + numRepeats ;
 

    if(m_global_instance_enabled)
    {
        LOG(LEVEL) << "[ creating extra mm --global_instance_enabled " ;
        bool global_instance = true ; 
        GMergedMesh* mmgi = m_geolib->makeMergedMesh( numRidx, base, root, verbosity, global_instance ); 
        mmgi->addInstancedBuffers(placements);  // call for global for common structure 
        LOG(LEVEL) << "] creating extra mm --global_instance_enabled " ;
    } 
    else
    {
        LOG(LEVEL) << "NOT creating extra mm as no  --global_instance_enabled " ;
    }


    LOG(info) 
        << " numRepeats " << numRepeats
        << " numRidx " << numRidx
        << " --global_instance_enabled " << m_global_instance_enabled 
        ;

    for(unsigned ridx=1 ; ridx < numRidx ; ridx++)  // 1-based index
    {
         const GNode*   rbase  = last ? getLastRepeatExample(ridx)  : getRepeatExample(ridx) ;  

         if(m_verbosity > 2)
         LOG(info) 
             << " ridx " << ridx 
             << " rbase " << rbase
             ;

         GMergedMesh* mm = m_geolib->makeMergedMesh(ridx, rbase, root, verbosity, false ); 

         std::vector<const GNode*> placements_ = getPlacements(ridx);

         mm->addInstancedBuffers(placements_);
     
         //mm->reportMeshUsage( ggeo, "GInstancer::CreateInstancedMergedMeshes reportMeshUsage (instanced)");
    }


}



/**
GInstancer::dumpMeshset
-------------------------

Dumping the unique LVs in each repeater

**/

void GInstancer::dumpMeshset() const 
{
    unsigned numRepeats = getNumRepeats();
    unsigned numRidx = numRepeats + 1 ; 
 
    LOG(info) 
        << " numRepeats " << numRepeats 
        << " numRidx " << numRidx
        << " (slot 0 for global non-instanced) "
        ;

    typedef std::set<unsigned> SU ; 

    for(unsigned ridx=0 ; ridx < numRidx ; ridx++ )
    {
        if( m_meshset.find(ridx) == m_meshset.end() ) continue ;   

        const SU& ms = m_meshset.at(ridx); 

        std::cout << " ridx " << ridx 
                  << " ms " << ms.size()
                  << " ( " 
                  ;
 
        for(SU::const_iterator it=ms.begin() ; it != ms.end() ; it++ )
              std::cout << *it << " " ;

        std::cout << " ) " << std::endl ; 

    }
}




void GInstancer::dumpCSGSkips() const 
{
    LOG(info) ;
    for( MUVU::const_iterator i = m_csgskiplv.begin() ; i != m_csgskiplv.end() ; i++ )
    {
        unsigned lvIdx = i->first ; 
        const VU& v = i->second ; 

        std::cout << " lvIdx " << lvIdx 
                  << " skip total : " << v.size()
                  << " nodeIdx ( " 
                  ; 

        unsigned nj = v.size() ; 


        //for(VU::const_iterator j=v.begin() ; j != v.end() ; j++ ) std::cout << *j << " " ;

        for( unsigned j=0 ; j < std::min( nj, 20u ) ; j++ ) std::cout << v[j] << " " ;  
        if( nj > 20u ) std::cout << " ... " ; 
        std::cout << " ) " << std::endl ; 
    }  
}

void GInstancer::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    dumpMeshset();
    dumpCSGSkips(); 
}






