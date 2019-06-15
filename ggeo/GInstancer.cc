
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


#include "PLOG.hh"


// Following enabling of vertex de-duping (done at AssimpWrap level) 
// the below criteria are finding fewer repeats, for DYB only the hemi pmt
// TODO: retune


const plog::Severity GInstancer::LEVEL = debug ; 

GInstancer::GInstancer(Opticks* ok, GGeoLib* geolib, GNodeLib* nodelib, NSceneConfig* config) 
    : 
    m_log(new SLog("GInstancer::GInstancer","", verbose)),
    m_ok(ok),
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
    m_globals_count(0)
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

Canonical invokation from GGeo::prepareMeshes

**/

void GInstancer::createInstancedMergedMeshes(bool delta, unsigned verbosity)
{
    BTimeKeeper t("GInstancer::createInstancedMergedMeshes") ; 
    t.setVerbose(true);
    t.start();

    if(delta) 
    {
        deltacheck();
        t("deltacheck"); 
    }

    traverse();   // spin over tree counting up progenyDigests to find repeated geometry 
    t("traverse"); 

    labelTree();  // recursive setRepeatIndex on the GNode tree for each of the repeated bits of geometry
    t("labelTree"); 


    LOG(LEVEL) << "( makeMergedMeshAndInstancedBuffers " ; 
    makeMergedMeshAndInstancedBuffers(verbosity);
    t("makeMergedMeshAndInstancedBuffers"); 
    LOG(LEVEL) << ") makeMergedMeshAndInstancedBuffers " ; 


    t.stop();

    if(t.deltaTime() > 0.1)
    t.dump("GInstancer::createInstancedMergedMeshes deltaTime > cut ");
}





void GInstancer::traverse()
{
    m_root = m_nodelib->getVolume(0);
    assert(m_root);

    // count occurences of distinct progeny digests (relative sub-tree identities) in m_digest_count 
    traverse_r(m_root, 0);

    m_digest_count->sort(false);   // descending count order, ie most common subtrees first
    //m_digest_count->dump();

    // minrep 120 removes repeats from headonPMT, calibration sources and RPC leaving just PMTs 
   
    // collect digests of repeated pieces of geometry into  m_repeat_candidates
    findRepeatCandidates(m_repeat_min, m_vertex_min); 

    unsigned num_reps = getNumRepeats();
    if(num_reps > 0 )
    dumpRepeatCandidates(20u);
}

void GInstancer::traverse_r( GNode* node, unsigned int depth)
{
    std::string& pdig = node->getProgenyDigest();
    m_digest_count->add(pdig.c_str());
    m_count++ ; 

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1 );
}


void GInstancer::deltacheck()
{
    // check consistency of the level transforms
    m_root = m_nodelib->getVolume(0);
    assert(m_root);

    deltacheck_r(m_root, 0);
}

void GInstancer::deltacheck_r( GNode* node, unsigned int depth)
{
    GVolume* volume = dynamic_cast<GVolume*>(node) ;
    GMatrixF* gtransform = volume->getTransform();

    // volumes levelTransform is set in AssimpGGeo and hails from the below with level -2
    //      aiMatrix4x4 AssimpNode::getLevelTransform(int level)
    //  looks to correspond to the placement of the LV within its PV  

    //GMatrixF* ltransform = volume->getLevelTransform();  
    GMatrixF* ctransform = volume->calculateTransform();
    float delta = gtransform->largestDiff(*ctransform);

    
    unsigned int nprogeny = node->getLastProgenyCount() ;

    if(nprogeny > 0 ) 
            LOG(debug) 
              << "GInstancer::deltacheck " 
              << " #progeny "  << std::setw(6) << nprogeny 
              << " delta*1e6 " << std::setprecision(6) << std::fixed << delta*1e6 
              << " name " << node->getName() 
              ;

    assert(delta < 1e-6) ;

    for(unsigned int i = 0; i < node->getNumChildren(); i++) deltacheck_r(node->getChild(i), depth + 1 );
}





struct GRepeat
{
    unsigned   repeat_min ; 
    unsigned   vertex_min ; 
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
          nprog(first->getLastProgenyCount()),
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



// suspect problem with allowing leaf repeaters is that digesta are not-specific enough, 
// so get bad matching 
//
//  allowing leaf repeaters results in too many, so place vertex count reqirement too 


void GInstancer::findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min)
{
    unsigned int nall = m_digest_count->size() ; 
    std::vector<GRepeat> cands ; 

    // over distinct subtrees (ie progeny digests)
    for(unsigned int i=0 ; i < nall ; i++)
    {
        std::pair<std::string,unsigned int>&  kv = m_digest_count->get(i) ;

        std::string& pdig = kv.first ; 
        unsigned int ndig = kv.second ;  // number of occurences of the progeny digest 

        GNode* first = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest

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
        unsigned dmax = 20u ;  
        LOG(info) 
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

bool GInstancer::isContainedRepeat( const std::string& pdig, unsigned int levels ) const 
{
    // for the first node that matches the *pdig* progeny digest
    // look back *levels* ancestors to see if any of the immediate ancestors 
    // are also repeat candidates, if they are then this is a contained repeat
    // and is thus disallowed in favor of the ancestor that contains it 

    GNode* node = m_root->findProgenyDigest(pdig) ; 
    std::vector<GNode*>& ancestors = node->getAncestors();  // ordered from root to parent 
    unsigned int asize = ancestors.size(); 

    for(unsigned int i=0 ; i < std::min(levels, asize) ; i++)
    {
        GNode* a = ancestors[asize - 1 - i] ; // from back to start with parent
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

    GNode* first = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    std::vector<GNode*> placements = m_root->findAllProgenyDigest(pdig);
    std::cout  
                  << " pdig "  << std::setw(32) << pdig  
                  << " ndig "  << std::setw(6) << std::dec << ndig
                  << " nprog " <<  std::setw(6) << std::dec << first->getLastProgenyCount() 
                  << " placements " << std::setw(6) << placements.size()
                  << " n "          <<  first->getName() 
                  << std::endl 
                  ;  

    assert(placements.size() == ndig ); // restricting traverse to just selected causes this to fail
    if(verbose)
    {
        for(unsigned int i=0 ; i < placements.size() ; i++)
        {
            GNode* place = placements[i] ;
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



void GInstancer::labelTree()   // hmm : doesnt label global volumes ?
{
    m_labels = 0 ; 

    for(unsigned int i=0 ; i < m_repeat_candidates.size() ; i++)
    {
         std::string pdig = m_repeat_candidates[i];
         unsigned int ridx = getRepeatIndex(pdig);
         assert(ridx == i + 1 );
         std::vector<GNode*> placements = m_root->findAllProgenyDigest(pdig);

         // recursive labelling starting from the placements
         for(unsigned int p=0 ; p < placements.size() ; p++)
         {
             labelRepeats_r(placements[p], ridx);
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
}

void GInstancer::labelRepeats_r( GNode* node, unsigned int ridx)
{
    node->setRepeatIndex(ridx);
    m_repeats_count += 1 ; 

    unsigned lvidx = node->getMeshIndex();  
    m_meshset[ridx].insert( lvidx ) ; 

    if( m_ok->isCSGSkipLV(lvidx) )   // --csgskiplv
    {
        GVolume* vol = dynamic_cast<GVolume*>(node); 
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
    for(unsigned int i = 0; i < node->getNumChildren(); i++) labelRepeats_r(node->getChild(i), ridx );
}


void GInstancer::traverseGlobals_r( GNode* node, unsigned depth )
{
    unsigned ridx = node->getRepeatIndex() ; 
    if( ridx > 0 ) return ; 
    // only recurse whilst in global territory with ridx == 0, as soon as hit a repeated volume stop recursing 
    assert( ridx == 0 ); 
    m_globals_count += 1 ; 

    unsigned lvidx = node->getMeshIndex();  
    m_meshset[ridx].insert( lvidx ) ; 

    if( m_ok->isCSGSkipLV(lvidx) )   // --csgskiplv
    {
        GVolume* vol = dynamic_cast<GVolume*>(node); 
        vol->setCSGSkip(true);      

        // Currently trying to skip global lv at this rather late juncture 
        // leads to inconsistencies manifesting in a corrupted color buffer 
        // (i recall that all global volumes are retained for index consistency in the merge of GMergedMesh GGeoLib)
        // so moved to simply editing the input GDML
        // Presumably could also do this by moving the skip earlier to the Geant4 X4 traverse
        // see notes/issues/torus_replacement_on_the_fly.rst

        m_csgskiplv[lvidx].push_back( node->getIndex() ); 
        m_csgskiplv_count += 1 ; 
    }
   
    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverseGlobals_r(node->getChild(i), depth + 1 );
}






std::vector<GNode*> GInstancer::getPlacements(unsigned int ridx)
{
    std::vector<GNode*> placements ;
    if(ridx == 0)
    {
        placements.push_back(m_root);
    }
    else
    {
        assert(ridx >= 1); // ridx is a 1-based index
        assert(ridx-1 < m_repeat_candidates.size()); 
        std::string pdig = m_repeat_candidates[ridx-1];
        placements = m_root->findAllProgenyDigest(pdig);
    } 
    return placements ; 
}

GNode* GInstancer::getRepeatExample(unsigned int ridx)
{
    std::vector<GNode*> placements = getPlacements(ridx);
    std::string pdig = m_repeat_candidates[ridx-1];
    GNode* node = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    assert(placements[0] == node);
    return node ; 
}


/**
GInstancer::makeMergedMeshAndInstancedBuffers
----------------------------------------------

Populates m_geolib with merged meshes including the instancing buffers.

**/

void GInstancer::makeMergedMeshAndInstancedBuffers(unsigned verbosity)
{

    GNode* root = m_nodelib->getNode(0);
    assert(root); 
    GNode* base = NULL ; 


    // passes thru to GMergedMesh::create with management of the mm in GGeoLib
    GMergedMesh* mm0 = m_geolib->makeMergedMesh(0, base, root, verbosity );


    std::vector<GNode*> placements = getPlacements(0);  // just m_root
    assert(placements.size() == 1 );
    mm0->addInstancedBuffers(placements);  // call for global for common structure 


    unsigned numRepeats = getNumRepeats();
    unsigned numRidx = numRepeats + 1 ; 
 
    LOG(info) 
              << " numRepeats " << numRepeats
              << " numRidx " << numRidx
              ;

    for(unsigned int ridx=1 ; ridx < numRidx ; ridx++)  // 1-based index
    {
         GNode*   rbase  = getRepeatExample(ridx) ;    // <--- why not the parent ? off-by-one confusion here as to which transforms to include

         if(m_verbosity > 2)
         LOG(info) 
                   << " ridx " << ridx 
                   << " rbase " << rbase
                   ;

         GMergedMesh* mm = m_geolib->makeMergedMesh(ridx, rbase, root, verbosity ); 

         std::vector<GNode*> placements_ = getPlacements(ridx);

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






