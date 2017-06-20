#include <sstream>
#include <iomanip>

#include "BFile.hh"

#include "NYGLTF.hpp"

#include "Counts.hpp"
#include "NTrianglesNPY.hpp"
#include "NParameters.hpp"
#include "NPY.hpp"
#include "NScene.hpp"
#include "NTxt.hpp"
#include "NCSG.hpp"
#include "NGLMExt.hpp"
#include "Nd.hpp"

#include "PLOG.hh"




nd* NScene::getNd(unsigned idx)
{
    return m_nd[idx] ; 
}
nd* NScene::getRoot()
{
    return m_root ; 
}
NCSG* NScene::getCSG(unsigned mesh_idx)
{
    return m_csg[mesh_idx];
}

bool NScene::Exists(const char* base, const char* name)
{
    return BFile::ExistsFile(base, name);
}


NScene* NScene::Load( const char* gltfbase, const char* gltfname, const char* gltfconfig) 
{
    NScene* scene =  NScene::Exists(gltfbase, gltfname) ? new NScene(gltfbase, gltfname, gltfconfig) : NULL ;
    if(!scene)
        LOG(fatal) << "NScene:Load MISSING PATH" 
                   << " gltfbase " << gltfbase
                   << " gltfname " << gltfname
                   << " gltfconfig " << gltfconfig
                   ; 

    return scene ; 
}


NScene::NScene(const char* base, const char* name, const char* config, int scene_idx)  
   :
    NGLTF(base, name, config, scene_idx),
    m_verbosity(0),
    m_num_global(0),
    m_num_csgskip(0),
    m_num_placeholder(0),
    m_csgskip_lvlist(NULL),
    m_placeholder_lvlist(NULL),
    m_node_count(0),
    m_label_count(0),
    m_digest_count(new Counts<unsigned>("progenyDigest"))
{
    init_lvlists(base, name);

    load_asset_extras();
    load_csg_metadata();

    m_root = import_r(0, NULL, 0); 

    if(m_verbosity > 1)
    dumpNdTree("NScene::NScene");

    compare_trees();

    count_progeny_digests();

    find_repeat_candidates();

    dump_repeat_candidates();

    labelTree();

    if(m_verbosity > 1)
    dumpRepeatCount(); 

    markGloballyUsedMeshes_r(m_root);

    // move load_mesh_extras later so can know which meshes are non-instanced needing 
    // gtransform slots for all primitives
    load_mesh_extras();

    write_lvlists();

    LOG(info) << "NScene::NScene DONE" ;  

    //assert(0 && "hari kari");

}

void NScene::init_lvlists(const char* base, const char* name)
{
    std::string stem = BFile::Stem(name);
    std::string csgskip_path = BFile::FormPath(base, stem.c_str(), "CSGSKIP_DEEP_TREES.txt");
    std::string placeholder_path = BFile::FormPath(base, stem.c_str(), "PLACEHOLDER_FAILED_POLY.txt");

    BFile::preparePath( csgskip_path.c_str(), true);
    BFile::preparePath( placeholder_path.c_str(), true);

    m_csgskip_lvlist     = new NTxt(csgskip_path.c_str());
    m_placeholder_lvlist = new NTxt(placeholder_path.c_str());

    LOG(info) << "NScene::init_lvlists" ;

    std::cout << " csgskip " << m_csgskip_lvlist->desc() << std::endl ; 
    std::cout << " placeholder(polyfail) " << m_placeholder_lvlist->desc() << std::endl ; 

}

void NScene::write_lvlists()
{
    LOG(info) << "NScene::write_lvlists";
    std::cout << " csgskip " << m_csgskip_lvlist->desc() << std::endl ; 
    std::cout << " placeholder(polyfail) " << m_placeholder_lvlist->desc() << std::endl ; 

    m_csgskip_lvlist->write();
    m_placeholder_lvlist->write();
}


void NScene::load_asset_extras()
{
    auto extras = m_gltf->asset.extras ; 
    m_verbosity = extras["verbosity"]; 

    LOG(info) << "NScene::load_asset_extras"
              << " m_verbosity " << m_verbosity 
               ;

}

unsigned NScene::getVerbosity()
{
    return m_verbosity ; 
}

void NScene::load_csg_metadata()
{
   // for debugging need the CSG metadata prior to loading the trees
    unsigned num_meshes = getNumMeshes();
    LOG(info) << "NScene::load_csg_metadata"
              << " num_meshes " << num_meshes
              ;

    for(std::size_t mesh_id = 0; mesh_id < num_meshes; ++mesh_id)
    {
        ygltf::mesh_t* mesh = getMesh(mesh_id);
        auto extras = mesh->extras ; 
        std::string uri = extras["uri"] ; 
        std::string csgpath = BFile::FormPath(m_base, uri.c_str() );

        NParameters* meta = NCSG::LoadMetadata(csgpath.c_str());

        m_csg_metadata[mesh_id] = meta ; 

        //std::cout << meshmeta(mesh_id) << std::endl ;
    }
}


// metadata from the root nodes of the CSG trees for each solid
// pmt-cd treebase.py:Node._get_meta
//
template<typename T>
T NScene::getCSGMeta(unsigned mesh_id, const char* key, const char* fallback ) const 
{
    const NParameters* meta = m_csg_metadata.at(mesh_id) ;   // operator[] can change the map if no such key
    return meta->get<T>(key, fallback) ;
}

template NPY_API std::string NScene::getCSGMeta<std::string>(unsigned,const char*, const char*) const ;
template NPY_API int         NScene::getCSGMeta<int>(unsigned,const char*, const char*) const ;
template NPY_API float       NScene::getCSGMeta<float>(unsigned,const char*, const char*) const ;
template NPY_API bool        NScene::getCSGMeta<bool>(unsigned,const char*, const char*) const ;

// keys need to match analytic/sc.py 
std::string NScene::lvname(unsigned mesh_id) const { return getCSGMeta<std::string>(mesh_id,"lvname","-") ; }
std::string NScene::soname(unsigned mesh_id) const { return getCSGMeta<std::string>(mesh_id,"soname","-") ; }
int         NScene::height(unsigned mesh_id) const { return getCSGMeta<int>(mesh_id,"height","-1") ; }
int         NScene::nchild(unsigned mesh_id) const { return getCSGMeta<int>(mesh_id,"nchild","-1") ; }
bool        NScene::isSkip(unsigned mesh_id) const { return getCSGMeta<int>(mesh_id,"skip","0") == 1 ; }

std::string NScene::meshmeta(unsigned mesh_id) const 
{
    std::stringstream ss ; 

    ss << " mesh_id " << std::setw(3) << mesh_id
       << " height "  << std::setw(2) << height(mesh_id)
       << " soname "  << std::setw(35) << soname(mesh_id)
       << " lvname "  << std::setw(35) << lvname(mesh_id)
       ;

    return ss.str();
}



void NScene::load_mesh_extras()
{
    unsigned num_meshes = getNumMeshes();
    assert( num_meshes == m_gltf->meshes.size() ); 

    LOG(info) << "NScene::load_mesh_extras START" 
              << " m_verbosity " << m_verbosity
              << " num_meshes " << num_meshes 
              ; 

    for(std::size_t mesh_id = 0; mesh_id < num_meshes; ++mesh_id)
    {
        //auto mesh = &m_gltf->meshes.at(mesh_id);
        //ygltf::mesh_t* mesh = &m_gltf->meshes.at(mesh_id);
        ygltf::mesh_t* mesh = getMesh(mesh_id);

        auto primitives = mesh->primitives ; 
        auto extras = mesh->extras ; 

        bool iug = isUsedGlobally(mesh_id); 
        if(iug) m_num_global++ ; 

        std::string uri = extras["uri"] ; 
        std::string csgpath = BFile::FormPath(m_base, uri.c_str() );

        int verbosity = 0 ; 
        bool polygonize = true ; 

        NCSG* csg = NCSG::LoadTree(csgpath.c_str(), iug, verbosity, polygonize  ); 
        csg->setIndex(mesh_id);

        bool csgskip = csg->isSkip() ;
        if(csgskip) 
        {
            m_csgskip_lvlist->addLine(mesh->name);
            m_num_csgskip++ ; 
            LOG(warning) << "NScene::load_mesh_extras"
                         << " csgskip CSG loaded " << csg->meta()
                          ;
        }

        NTrianglesNPY* tris = csg->getTris();
        assert(tris);
        bool placeholder = tris->isPlaceholder();
        if(placeholder) 
        {
            m_placeholder_lvlist->addLine(mesh->name);
            m_num_placeholder++ ; 
        }


        m_csg[mesh_id] = csg ; 

        std::cout << " mId " << std::setw(4) << mesh_id 
                  << " npr " << std::setw(4) << primitives.size() 
                  << " nam " << std::setw(65) << mesh->name 
                  << " iug " << std::setw(1) << iug 
                  << " poly " << std::setw(3) << tris->getPoly()
                  << " smry " << csg->smry() 
                  << std::endl ; 
    }  


    LOG(info) << "NScene::load_mesh_extras DONE"
              << " m_verbosity " << m_verbosity
              << " num_meshes " << num_meshes
              << " m_num_global " << m_num_global
              << " m_num_csgskip " << m_num_csgskip
              << " m_num_placeholder " << m_num_placeholder
              ;



}





nd* NScene::import_r(int idx,  nd* parent, int depth)
{
    ygltf::node_t* ynode = getNode(idx);
    auto extras = ynode->extras ; 
    std::string boundary = extras["boundary"] ; 
 
    nd* n = new nd ;   // NB these are structural nodes, not CSG tree nodes

    n->idx = idx ; 
    n->repeatIdx = 0 ; 
    n->mesh = ynode->mesh ; 
    n->parent = parent ;
    n->depth = depth ;
    n->boundary = boundary ;
    n->transform = new nmat4triple( ynode->matrix.data() ); 
    n->gtransform = nd::make_global_transform(n) ;   

    for(int child : ynode->children) n->children.push_back(import_r(child, n, depth+1));  // recursive call

    m_nd[idx] = n ;

    return n ; 
}




void NScene::count_progeny_digests_r(nd* n)
{
    const std::string& pdig = n->get_progeny_digest();

    m_digest_count->add(pdig.c_str());
    m_node_count++ ; 

    for(nd* c : n->children) count_progeny_digests_r(c) ;
}

void NScene::count_progeny_digests()
{
    count_progeny_digests_r(m_root);

    m_digest_count->sort(false);   // descending count order, ie most common subtrees first

    bool dump = false ;

    if(dump)
    m_digest_count->dump();

    LOG(info) << "NScene::count_progeny_digests"
              << " node_count " << m_node_count 
              << " digest_size " << m_digest_count->size()
              ;  

    for(unsigned i=0 ; i < m_digest_count->size() ; i++)
    {
        const std::pair<std::string, unsigned>&  su =  m_digest_count->get(i);
        std::string pdig = su.first ;
        unsigned num = su.second ;  

        std::vector<nd*> selection = m_root->find_nodes(pdig);
        assert( selection.size() == num );

        nd* first = m_root->find_node(pdig);

        if(num > 0)
        {
            assert( first && selection[0] == first );
        }

        assert(first);
       
        unsigned mesh_id = first->mesh ; 

        if(dump)
        std::cout << " pdig " << std::setw(32) << pdig
                  << " num " << std::setw(5) << num
                  << " meshmeta " << meshmeta(mesh_id)
                  << std::endl 
                ; 
    } 
}


void NScene::find_repeat_candidates()
{
   // hmm : this approach will not gang together siblings 
   //       that always appear together, only subtrees 

    unsigned repeat_min = 4 ; 

    unsigned int num_progeny_digests = m_digest_count->size() ;

    LOG(debug) << "NScene::find_repeat_candidates"
              << " num_progeny_digests " << num_progeny_digests 
              << " candidates marked with ** "
              ;   

    for(unsigned i=0 ; i < num_progeny_digests ; i++)
    {   
        std::pair<std::string,unsigned int>&  kv = m_digest_count->get(i) ;

        std::string& pdig = kv.first ; 
        unsigned int num_pdig = kv.second ;   

        bool select = num_pdig > repeat_min ;

/*
        nd* first = m_root->find_node(pdig) ;
        unsigned num_progeny = first->get_progeny_count() ;  // includes self 
        std::cout  
                  << ( select ? "**" : "  " ) 
                  << " i "         << std::setw(3) << i 
                  << " pdig "      << std::setw(32) << pdig  
                  << " num_pdig "  << std::setw(6) << num_pdig
                  << " num_progeny "     <<  std::setw(6) << num_progeny
                  << " meshmeta "  <<  meshmeta(first->mesh)
                  << std::endl 
                  ;
*/

        if(select) m_repeat_candidates.push_back(pdig);
    }

    // erase repeats that are enclosed within other repeats 
    // ie that have an ancestor which is also a repeat candidate

    m_repeat_candidates.erase(
         std::remove_if(m_repeat_candidates.begin(), m_repeat_candidates.end(), *this ),
         m_repeat_candidates.end()
    );


}

bool NScene::operator()(const std::string& pdig)
{
    bool cr = is_contained_repeat(pdig, 3);

/*
    if(cr) LOG(info)
                  << "NScene::operator() "
                  << " pdig "  << std::setw(32) << pdig
                  << " disallowd as is_contained_repeat "
                  ;
*/

    return cr ;
}


bool NScene::is_contained_repeat( const std::string& pdig, unsigned levels ) 
{
    // for the first node that matches the *pdig* progeny digest
    // look back *levels* ancestors to see if any of the immediate ancestors 
    // are also repeat candidates, if they are then this is a contained repeat
    // and is thus disallowed in favor of the ancestor that contains it 

    nd* n = m_root->find_node(pdig) ;
    const std::vector<nd*>& ancestors = n->get_ancestors();  // ordered from root to parent 
    unsigned int asize = ancestors.size();

    for(unsigned i=0 ; i < std::min(levels, asize) ; i++)
    {
        nd* a = ancestors[asize - 1 - i] ; // from back to start with parent
        const std::string& adig = a->get_progeny_digest();

        if(std::find(m_repeat_candidates.begin(), m_repeat_candidates.end(), adig ) != m_repeat_candidates.end())
        {
            return true ;
        }
    }
    return false ;
}

void NScene::dump_repeat_candidates() const
{
    unsigned num_repeat_candidates = m_repeat_candidates.size() ;
    LOG(info) << "NScene::dump_repeat_candidates"
              << " num_repeat_candidates " << num_repeat_candidates 
              ;
    for(unsigned i=0 ; i < num_repeat_candidates ; i++)
        dump_repeat_candidate(i);
} 

void NScene::dump_repeat_candidate(unsigned idx) const 
{
    std::string pdig = m_repeat_candidates[idx];
    unsigned num_instances = m_digest_count->getCount(pdig.c_str());

    nd* first = m_root->find_node(pdig) ;
    assert(first);

    unsigned num_progeny = first->get_progeny_count() ;  
    unsigned mesh_id = first->mesh ; 

    std::vector<nd*> placements = m_root->find_nodes(pdig);
    assert( placements[0] == first );
    assert( placements.size() == num_instances );

    for(unsigned i=0 ; i < num_instances ; i++)
    {
       assert( placements[i]->get_progeny_count() == num_progeny ) ; 
       assert( placements[i]->mesh == mesh_id ) ; 
    }

    std::cout
              << " idx "    << std::setw(3) << idx 
              << " pdig "   << std::setw(32) << pdig
              << " nprog "  << std::setw(5) << num_progeny
              << " ninst " << std::setw(5) << num_instances
              << " mmeta " << meshmeta(mesh_id)
              << std::endl 
              ; 

    bool verbose = num_progeny > 0 ; 

    if(verbose)
    {
        const std::vector<nd*>& progeny = first->get_progeny();    
        assert(num_progeny == progeny.size());
        for(unsigned p=0 ; p < progeny.size() ; p++)
        {
            
            std::cout << "(" << std::setw(2) << p << ") "  
                      << meshmeta(progeny[p]->mesh)
                      << std::endl ; 
        }
    }

 
}




void NScene::dumpNdTree(const char* msg)
{
    LOG(info) << msg << " m_verbosity " << m_verbosity  ; 
    dumpNdTree_r( m_root ) ; 
}
void NScene::dumpNdTree_r(nd* n)
{
    std::cout << n->desc() << std::endl ;  

    if(m_verbosity > 3)
    {
        if(n->transform)   
        std::cout <<  "n.transform  " << *n->transform << std::endl ; 

        if(n->gtransform)   
        std::cout <<  "n.gtransform " << *n->gtransform << std::endl ; 
    }

    for(nd* cn : n->children) dumpNdTree_r(cn) ;
}



void NScene::compare_trees()
{
    compare_trees_r(0);
}

void NScene::compare_trees_r(int idx)
{
    ygltf::node_t* ynode = getNode(idx);
    nd* n = getNd(idx);    

    assert( ynode->mesh == int(n->mesh) );
    assert( ynode->children.size() == n->children.size() );
    assert( n->transform) ; 
    assert( n->gtransform) ; 

    {
        std::array<float,16> tt ;   
        nglmext::copyTransform(tt, n->transform->t );

        bool local_match = tt == ynode->matrix ; 
        if(!local_match)
        {
            std::cout << "idx " << idx << ( local_match ? " LOCAL-MATCH " : " LOCAL-MISMATCH " ) << std::endl ; 
            std::cout << "ntt " << nglmext::xform_string( tt ) << std::endl ;    
            std::cout << "nmx " << nglmext::xform_string( ynode->matrix ) << std::endl ;    
        }
        assert(local_match);
    }


    {
        std::array<float,16> gg ;   
        nglmext::copyTransform(gg, n->gtransform->t );
        const std::array<float,16>& fxf = getFlatTransform(idx) ; 
        const std::array<float,16>& nxf = getNodeTransform(idx) ; 

        bool global_match = gg == fxf && fxf == nxf  ; 
        if(!global_match)
        {
            std::cout << "idx " << idx << ( global_match ? " GLOBAL-MATCH " : " GLOBAL-MISMATCH " ) << std::endl ; 
            std::cout << "gg  " << nglmext::xform_string( gg ) << std::endl ;    
            std::cout << "fxf " << nglmext::xform_string( fxf ) << std::endl ;    
            std::cout << "nxf " << nglmext::xform_string( nxf ) << std::endl ;    
        }
        assert(global_match);
    }


    for(int child : ynode->children) compare_trees_r( child );
}



unsigned NScene::deviseRepeatIndex_0(nd* n)
{
    unsigned mesh_idx = n->mesh ; 
    unsigned num_mesh_instances = getNumInstances(mesh_idx) ;

    unsigned ridx = 0 ;   // <-- global default ridx

    bool make_instance  = num_mesh_instances > 4  ;

    if(make_instance)
    {
        if(m_mesh2ridx.count(mesh_idx) == 0)
             m_mesh2ridx[mesh_idx] = m_mesh2ridx.size() + 1 ;

        ridx = m_mesh2ridx[mesh_idx] ;

        // ridx is a 1-based contiguous index tied to the mesh_idx 
        // using trivial things like "mesh_idx + 1" causes  
        // issue downstream which expects a contiguous range of ridx 
        // when using partial geometries 
    }
    return ridx ;
}


unsigned NScene::deviseRepeatIndex(const std::string& pdig )
{
    // repeat index corresponding to a digest
     unsigned ridx(0);
     std::vector<std::string>::iterator it = std::find(m_repeat_candidates.begin(), m_repeat_candidates.end(), pdig );
     if(it != m_repeat_candidates.end())
     {
         ridx = 1 + std::distance(m_repeat_candidates.begin(), it ) ;  // 1-based index
         LOG(debug)<<"NScene::deviseRepeatIndex "
                  << std::setw(32) << pdig
                  << " ridx " << ridx
                  ;
     }
     return ridx ;
}


void NScene::labelTree()
{
    for(unsigned i=0 ; i < m_repeat_candidates.size() ; i++)
    {
         std::string pdig = m_repeat_candidates.at(i);

         unsigned ridx = deviseRepeatIndex(pdig);

         assert(ridx == i + 1 );

         std::vector<nd*> instances = m_root->find_nodes(pdig);

         // recursive labelling starting from the instances
         for(unsigned int p=0 ; p < instances.size() ; p++)
         {
             labelTree_r(instances[p], ridx);
         }
    }

    LOG(info)<<"NScene::labelTree count of non-zero ridx labelTree_r " << m_label_count ;
}

#ifdef OLD_LABEL_TREE
void NScene::labelTree_r(nd* n, unsigned /*ridx*/)
{
    unsigned ridx = deviseRepeatIndex_0(n) ;
#else
void NScene::labelTree_r(nd* n, unsigned ridx)
{
#endif
    n->repeatIdx = ridx ;

    if(m_repeat_count.count(ridx) == 0) m_repeat_count[ridx] = 0 ; 
    m_repeat_count[ridx]++ ;

    if(ridx > 0) m_label_count++ ;  

    for(nd* c : n->children) labelTree_r(c, ridx) ;
}



void NScene::markGloballyUsedMeshes_r(nd* n)
{
    assert( n->repeatIdx > -1 );

    //if(n->repeatIdx == 0) setIsUsedGlobally(n->mesh, true );
    setIsUsedGlobally(n->mesh, true );

    // see opticks/notes/issues/subtree_instances_missing_transform.rst


    for(nd* c : n->children) markGloballyUsedMeshes_r(c) ;
}



void NScene::dumpRepeatCount()
{
    LOG(info) << "NScene::dumpRepeatCount" 
              << " m_verbosity " << m_verbosity 
               ; 

    typedef std::map<unsigned, unsigned> MUU ;
    unsigned totCount = 0 ;

    for(MUU::const_iterator it=m_repeat_count.begin() ; it != m_repeat_count.end() ; it++)
    {
        unsigned ridx = it->first ;
        unsigned count = it->second ;
        totCount += count ;
        std::cout
                  << " ridx " << std::setw(3) << ridx
                  << " count " << std::setw(5) << count
                  << std::endl ; 
    }
    LOG(info) << "NScene::dumpRepeatCount" 
              << " totCount " << totCount 
               ; 
}   

unsigned NScene::getRepeatCount(unsigned ridx)
{       
    return m_repeat_count[ridx] ; 
}   
unsigned NScene::getNumRepeats()
{       
   // this assumes ridx is a contiguous index
    return m_repeat_count.size() ;
}



