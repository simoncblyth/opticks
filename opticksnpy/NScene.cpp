#include <sstream>
#include <iomanip>

#include "BFile.hh"

#include "NYGLTF.hpp"

#include "NPY.hpp"
#include "NScene.hpp"
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



NScene::NScene(const char* base, const char* name, const char* config, int scene_idx)  
   :
    NGLTF(base, name, config, scene_idx),
    m_verbosity(0)
{
    load_asset_extras();

    m_root = import();

    if(m_verbosity > 1)
    dumpNdTree("NScene::NScene");

    compare_trees();

    labelTree_r(m_root);

    if(m_verbosity > 1)
    dumpRepeatCount(); 

    markGloballyUsedMeshes_r(m_root);

    // move load_mesh_extras later so can know which meshes are non-instanced needing 
    // gtransform slots for all primitives
    load_mesh_extras();

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


void NScene::load_mesh_extras()
{
    unsigned num_meshes = getNumMeshes();
    assert( num_meshes == m_gltf->meshes.size() ); 

    LOG(info) << "NScene::load_mesh_extras"
              << " num_meshes " << num_meshes
              ;


    for(std::size_t mesh_id = 0; mesh_id < num_meshes; ++mesh_id)
    {
        auto mesh = &m_gltf->meshes.at(mesh_id);

        auto primitives = mesh->primitives ; 
        auto extras = mesh->extras ; 

        bool iug = isUsedGlobally(mesh_id); 

        std::string uri = extras["uri"] ; 
        std::string csgpath = BFile::FormPath(m_base, uri.c_str() );

        int verbosity = 0 ; 
        bool polygonize = true ; 

        NCSG* csg = NCSG::LoadTree(csgpath.c_str(), iug, verbosity, polygonize  ); 
        csg->setIndex(mesh_id);

        m_csg[mesh_id] = csg ; 

        std::cout << " mid " << std::setw(4) << mesh_id 
                  << " prm " << std::setw(4) << primitives.size() 
                  << " nam " << std::setw(65) << mesh->name 
                  << " iug " << std::setw(1) << iug 
                  << " smry " << csg->smry() 
                  << std::endl ; 
    }  
}




nd* NScene::import()
{
    return import_r(0, NULL, 0); 
}

nd* NScene::import_r(int idx,  nd* parent, int depth)
{

    ygltf::node_t* ynode = getNode(idx);
    auto extras = ynode->extras ; 
    std::string boundary = extras["boundary"] ; 
 
    nd* n = new nd ;   // NB these are structural nodes, not CSG tree nodes

    n->idx = idx ; 
    n->repeatIdx = -1 ; 
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



unsigned NScene::deviseRepeatIndex(nd* n)
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

void NScene::labelTree_r(nd* n)
{
    unsigned ridx = deviseRepeatIndex(n);

    n->repeatIdx = ridx ;

    if(m_repeat_count.count(ridx) == 0) m_repeat_count[ridx] = 0 ; 
    m_repeat_count[ridx]++ ; 


    for(nd* c : n->children) labelTree_r(c) ;
}



void NScene::markGloballyUsedMeshes_r(nd* n)
{
    assert( n->repeatIdx > -1 );
    if(n->repeatIdx == 0) setIsUsedGlobally(n->mesh, true );

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



