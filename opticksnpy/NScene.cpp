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
    return m_csg_trees[mesh_idx];
}



NScene::NScene(const char* base, const char* name, int scene_idx)  
   :
    NGLTF(base, name, scene_idx)
{
    load_mesh_extras();
    import();
    compare_trees();
}

void NScene::load_mesh_extras()
{
    unsigned num_meshes = getNumMeshes();
    assert( num_meshes == m_gltf->meshes.size() ); 

    std::cout << "NScene::load_mesh_extras"
              << " num_meshes " << num_meshes
              << std::endl ; 


    for(std::size_t mesh_id = 0; mesh_id < num_meshes; ++mesh_id)
    {
        auto mesh = &m_gltf->meshes.at(mesh_id);

        auto primitives = mesh->primitives ; 
        auto extras = mesh->extras ; 

        std::string uri = extras["uri"] ; 
        std::string csgpath = BFile::FormPath(m_base, uri.c_str() );

        int verbosity = 0 ; 
        bool polygonize = true ; 
        NCSG* csg = NCSG::LoadTree(csgpath.c_str(), verbosity, polygonize  ); 
        csg->setIndex(mesh_id);

        m_csg_trees.push_back(csg); 

        std::cout << " mesh_id " << std::setw(4) << mesh_id 
                  << " primitives " << std::setw(4) << primitives.size() 
                  << " name " << std::setw(65) << mesh->name 
                  << " csgsmry " << csg->smry() 
                  << std::endl ; 
    }  
}




void NScene::import()
{
    m_root = import_r(0, NULL, 0); 
}

nd* NScene::import_r(int idx,  nd* parent, int depth)
{
    ygltf::node_t* node = getNode(idx);
    auto extras = node->extras ; 
    std::string boundary = extras["boundary"] ; 
 
    nd* n = new nd ; 

    n->idx = idx ; 
    n->mesh = node->mesh ; 
    n->parent = parent ;
    n->depth = depth ;
    n->boundary = boundary ;
    n->transform = new nmat4triple( node->matrix.data() ); 
    n->gtransform = nd::make_global_transform(n) ;   

    for(auto child : node->children) n->children.push_back(import_r(child, n, depth+1));  // recursive call

    m_nd[idx] = n ;

    return n ; 
}


void NScene::dumpNdTree(const char* msg)
{
    LOG(info) << msg ; 
    dumpNdTree_r( m_root ) ; 
}
void NScene::dumpNdTree_r(nd* n)
{
    std::cout << n->desc() << std::endl ;  
    for(auto cn : n->children) dumpNdTree_r(cn) ;
}



void NScene::compare_trees()
{
    compare_trees_r(0);
}

void NScene::compare_trees_r(int idx)
{
    ygltf::node_t* node = getNode(idx);
    nd* n = getNd(idx);    

    assert( node->mesh == int(n->mesh) );
    assert( node->children.size() == n->children.size() );
    assert( n->transform) ; 
    assert( n->gtransform) ; 

    {
        std::array<float,16> tt ;   
        nglmext::copyTransform(tt, n->transform->t );

        bool local_match = tt == node->matrix ; 
        if(!local_match)
        {
            std::cout << "idx " << idx << ( local_match ? " LOCAL-MATCH " : " LOCAL-MISMATCH " ) << std::endl ; 
            std::cout << "ntt " << nglmext::xform_string( tt ) << std::endl ;    
            std::cout << "nmx " << nglmext::xform_string( node->matrix ) << std::endl ;    
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


    for(auto child : node->children) compare_trees_r( child );
}

