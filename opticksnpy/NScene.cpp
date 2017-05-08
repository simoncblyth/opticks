#include <sstream>
#include <iomanip>

#include "BFile.hh"

#include "NYGLTF.hpp"

#include "NPY.hpp"
#include "NScene.hpp"
#include "NCSG.hpp"
#include "NGLMExt.hpp"

#include "PLOG.hh"



static inline std::string xform_string( const std::array<float, 16>& xform )
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < 16 ; i++) 
    {
        bool translation =  i == 12 || i == 13 || i == 14 ; 
        int fwid = translation ? 8 : 6 ;  
        int fprec = translation ? 2 : 3 ; 
        ss << std::setw(fwid) << std::fixed << std::setprecision(fprec) << xform[i] << ' ' ; 
    }
    return ss.str();
}


// Extracts from /usr/local/opticks/externals/yoctogl/yocto-gl/yocto/yocto_gltf.cpp

static inline std::array<float, 16> _float4x4_mul(
    const std::array<float, 16>& a, const std::array<float, 16>& b) 
{
    auto c = std::array<float, 16>();
    for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
            c[j * 4 + i] = 0;
            for (auto k = 0; k < 4; k++)
                c[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
        }
    }
    return c;
}

const std::array<float, 16> _identity_float4x4 = {{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};





NScene::NScene(const char* base, const char* name, int scene_idx)
   :
    m_base(strdup(base)),
    m_name(strdup(name)),
    m_scene_idx(scene_idx),
    m_gltf(NULL),
    m_fgltf(NULL)
{
    load();
    load_mesh_extras();

    import();
}




void NScene::load()
{
    std::string path = BFile::FormPath(m_base, m_name);

    LOG(info) << "NScene::load"
              << " path " << path
              ;

    bool load_bin = true ; 
    bool load_shaders = true ; 
    bool load_img = false ; 
    bool skip_missing = true  ;   

    m_gltf = ygltf::load_gltf(path, load_bin, load_shaders, load_img, skip_missing ) ;
    m_fgltf = ygltf::flatten_gltf(m_gltf, m_scene_idx); 

    collect_node_transforms();

    collect_mesh_instances();
    collect_mesh_totals();
}


void NScene::collect_node_transforms()
{
    auto scn = &m_gltf->scenes.at(m_scene_idx);

    auto stack = std::vector<std::tuple<int, std::array<float, 16>>>();

    for (auto node_id : scn->nodes)   // pushing root nodes, typically only one
    {
        stack.push_back(std::make_tuple(node_id, _identity_float4x4));
    }

    assert( stack.size() == 1);

    while (!stack.empty()) // stack based recursion
    {
        int              node_id;
        std::array<float, 16> xf;
        std::tie(node_id, xf)   = stack.back();
        stack.pop_back();   

        auto node = &m_gltf->nodes.at(node_id);
        xf = _float4x4_mul(xf, node_transform(node));   //   T-R-S-M    


        assert( m_xf.count(node_id) == 0 );
        assert( m_node2traversal.count(node_id) == 0 );

        m_node2traversal[node_id] = m_xf.size() ;  // <-- rosetta stone 
        m_xf[node_id] = xf ; 

        for (auto child : node->children) stack.push_back( {child, xf} ); 
    }
}







void NScene::collect_mesh_instances()
{
    assert( m_gltf->nodes.size() == m_fgltf->meshes.size() ); 
    // m_fgltf->meshes is bad name, more like nodes : as loads of them, not distinct shapes 

    for(std::size_t ifm = 0; ifm < m_fgltf->meshes.size(); ++ifm)  
    {
        auto fl_msh = m_fgltf->meshes[ifm] ;
        auto npr = fl_msh->primitives.size() ;
        assert(npr == 1);
        auto mesh_id = fl_msh->primitives[0] ;

        if(m_mesh_instances.count(mesh_id) == 0) m_mesh_instances[mesh_id] = {} ;  

        m_mesh_instances[mesh_id].push_back(ifm)  ;

    }

    assert( m_mesh_instances.size() == m_gltf->meshes.size()  );
}




void NScene::collect_mesh_totals(int scn_id)
{
    // TODO: fix, this just seeing root node

    auto scn = &m_gltf->scenes.at(scn_id);
    std::cout << "NScene::collect_mesh_totals"
              << " scn_id " << scn_id 
              << " scn->nodes.size() " << scn->nodes.size()
              << std::endl ; 

    for (auto node_id : scn->nodes) 
    {
        auto node = &m_gltf->nodes.at(node_id);

        std::cout << "NScene::collect_mesh_totals"
                  << " node_id " << node_id
                  << " mesh " << node->mesh 
                  << std::endl ; 

        if( m_mesh_totals.count(node->mesh) == 0) m_mesh_totals[node->mesh] = 0 ;  
        m_mesh_totals[node->mesh]++  ;
    }
}

void NScene::dump_mesh_totals(const char* msg)
{
    LOG(info) << msg ; 

    int node_total = 0 ; 

    typedef std::map<int, int> II ; 
    for(II::const_iterator it=m_mesh_totals.begin() ; it != m_mesh_totals.end() ; it++)
    {
       int mesh_id = it->first ;  
       int mesh_count = it->second ;  
       auto mesh = &m_gltf->meshes.at(mesh_id);
       node_total += mesh_count ; 
       std::cout 
             << std::setw(4) << mesh_id  
             << " : " 
             << std::setw(4) << mesh_count
             << " : "
             << mesh->name
             << std::endl ;
    }
    std::cout << " node_total " << node_total << std::endl ;  
}


void NScene::load_mesh_extras()
{
    std::cout << "NScene::load_mesh_extras"
              << " m_gltf->meshes.size() " << m_gltf->meshes.size()
              << std::endl ; 


    for(std::size_t mesh_id = 0; mesh_id < m_gltf->meshes.size(); ++mesh_id)
    {
        auto mesh = &m_gltf->meshes.at(mesh_id);

        auto primitives = mesh->primitives ; 

        auto extras = mesh->extras ; 

        // https://nlohmann.github.io/json/
        // TODO: handle non existing 
        std::string uri = extras["uri"] ; 
        //std::string soName = extras["soName"] ; 
        //int lvIdx = extras["lvIdx"] ; 

        std::string csgpath = BFile::FormPath(m_base, uri.c_str() );

        int verbosity = 0 ; 
        bool polygonize = true ; 
        NCSG* csg = NCSG::LoadTree(csgpath.c_str(), verbosity, polygonize  ); 
        m_csg_trees.push_back(csg); 

        std::cout << " mesh_id " << std::setw(4) << mesh_id 
                  << " primitives " << std::setw(4) << primitives.size() 
                  << " name " << std::setw(65) << mesh->name 
                  << " csgsmry " << csg->smry() 
                  << std::endl ; 

    }  
}









unsigned NScene::getNumMeshes()
{
    return m_mesh_instances.size() ;
}

unsigned NScene::getNumInstances(unsigned mesh_idx)
{
    assert( mesh_idx < m_mesh_instances.size() );
    return m_mesh_instances[mesh_idx].size() ; 
}

int NScene::getInstanceNodeIndex( unsigned mesh_idx, unsigned instance_idx)
{
    assert( mesh_idx < m_mesh_instances.size() );
    const std::vector<int>& instances = m_mesh_instances[mesh_idx] ;
    assert( instance_idx < instances.size() ); 
    int ifm = instances[instance_idx] ; 
    return ifm ; 
}



ygltf::fl_mesh* NScene::getFlatNode(unsigned node_idx)
{
    assert( m_node2traversal.count(node_idx) == 1 );
    unsigned traversal_idx = m_node2traversal[node_idx] ;  
    assert( traversal_idx < m_fgltf->meshes.size() );   
    // must use this rosetta stone as  m_fgltf->meshes in traversal order, not node_idx order
    ygltf::fl_mesh* fln = m_fgltf->meshes[traversal_idx] ;  
    return fln ; 
}

const std::array<float, 16>& NScene::getFlatTransform(unsigned node_idx )
{
    ygltf::fl_mesh* fln = getFlatNode(node_idx);
    return fln->xform ;
} 

const std::array<float,16>& NScene::getNodeTransform(unsigned node_idx)
{
    // yoctogl doesnt make it easy to look up the transform for a node
    // so collected m_xf in collect_node_transforms
    return m_xf[node_idx] ; 
}

glm::mat4 NScene::getTransformMatrix( unsigned node_idx )
{
    const std::array<float, 16>& nxf = getNodeTransform(node_idx);
    const std::array<float, 16>& fxf = getFlatTransform(node_idx);
    assert( nxf == fxf );
    return glm::make_mat4( nxf.data() );
}





NPY<float>* NScene::makeInstanceTransformsBuffer(unsigned mesh_idx)
{
    unsigned num_instances = getNumInstances(mesh_idx);
    NPY<float>* buf = NPY<float>::make(num_instances, 4, 4);
    buf->zero();

    for(unsigned i=0 ; i < num_instances ; i++)
    {
        int node_idx = getInstanceNodeIndex(mesh_idx, i );
        const std::array<float, 16>& xform = getNodeTransform(node_idx ); 

        for(unsigned j=0 ; j < 4 ; j++){
        for(unsigned k=0 ; k < 4 ; k++)
        {
            buf->setValue(i,j,k,0, xform[j*4+k]);
        }
        } 
    }

    if(mesh_idx == 0) 
    {
       assert(buf->getNumItems() == 1);
    }
    return buf ;
}




ygltf::node_t* NScene::getNode(unsigned node_idx)
{
    assert(node_idx < m_gltf->nodes.size() );   
    return &m_gltf->nodes.at(node_idx);
}  
std::string NScene::descNode( unsigned node_idx )
{
    ygltf::node_t* node = getNode(node_idx);
    std::stringstream ss ; 
    ss
           << "[" << std::setw(4) << node_idx  << "]"
           << " node.mesh " << std::setw(3) << node->mesh 
           << " node.nchild " << std::setw(3) << node->children.size() 
           << " node.matrix " << xform_string( node->matrix )  
           << " node.name " << node->name 
           ;
    return ss.str() ;
}




std::string NScene::descFlatNode( unsigned node_idx )
{
    ygltf::fl_mesh* fln = getFlatNode(node_idx);
    std::stringstream ss ; 
    ss
           << "[" << std::setw(4) << node_idx  << "]"
           << xform_string( fln->xform )  
           << fln->name 
           ;
    return ss.str() ;
}


 
std::string NScene::descInstance( unsigned mesh_idx, unsigned instance_idx )
{
    int node_idx = getInstanceNodeIndex(mesh_idx, instance_idx);
    std::stringstream ss ; 
    ss     
       << "[" << std::setw(3) << node_idx << ":" << std::setw(3) << instance_idx << "] "
       << descFlatNode(node_idx )
       ;
    return ss.str();
}
   
void NScene::dumpAllInstances( unsigned mesh_idx)
{
    assert( mesh_idx < m_mesh_instances.size() );
    unsigned num_instances = getNumInstances(mesh_idx) ;
    for(unsigned i=0 ; i < num_instances ; i++) std::cout << descInstance(mesh_idx, i) << std::endl ; 
}

void NScene::dumpAll()
{
    unsigned num_meshes = getNumMeshes() ;
    for(unsigned i=0 ; i < num_meshes ; i++) dumpAllInstances(i) ; 
}


void NScene::dump(const char* msg)
{
    LOG(info) << msg ; 

    std::cout << "glTF_t"
              << " scene " << m_gltf->scene
              << " accessors " << m_gltf->accessors.size()
              << " animations " << m_gltf->animations.size()
              << " bufferViews " << m_gltf->bufferViews.size()
              << " buffers " << m_gltf->buffers.size()
              << " cameras " << m_gltf->cameras.size()
              << " extensionsRequired " << m_gltf->extensionsRequired.size()
              << " extensionsUsed " << m_gltf->extensionsUsed.size()
              << " images " << m_gltf->images.size()
              << " materials " << m_gltf->materials.size()
              << " meshes " << m_gltf->meshes.size()
              << " nodes " << m_gltf->nodes.size()
              << " samplers " << m_gltf->samplers.size()
              << " scenes " << m_gltf->scenes.size()
              << " skins " << m_gltf->skins.size()
              << " textures " << m_gltf->textures.size()
              << std::endl ; 

    std::cout            
             << " default_scene " << m_fgltf->default_scene 
             << " fl_camera " << std::setw(3) << m_fgltf->cameras.size()
             << " fl_material " << std::setw(3) << m_fgltf->materials.size()
             << " fl_texture " << std::setw(3) << m_fgltf->textures.size()
             << " fl_primitives " << std::setw(3) << m_fgltf->primitives.size()
             << " fl_mesh " << std::setw(3) << m_fgltf->meshes.size()
             << " fl_scene " << std::setw(3) << m_fgltf->scenes.size()
             << std::endl ; 


    dump_mesh_totals();

}

void NScene::dump_scenes(const char* msg)
{
    LOG(info) << msg ; 
    for(auto fl_scn : m_fgltf->scenes)
         std::cout            
             << " camera " << std::setw(3) << fl_scn->cameras.size()
             << " material " << std::setw(3) << fl_scn->materials.size()
             << " texture " << std::setw(3) << fl_scn->textures.size()
             << " primitives " << std::setw(3) << fl_scn->primitives.size()
             << " mesh " << std::setw(3) << fl_scn->meshes.size()
             << " transform " << std::setw(3) << fl_scn->transforms.size()
             << std::endl ; 

}

void NScene::dump_flat_nodes(const char* msg)
{
    LOG(info) << msg << " " << m_fgltf->meshes.size() ; 

    for(auto fl_msh : m_fgltf->meshes)
    {
        auto npr = fl_msh->primitives.size() ;
        std::cout << "pr:" << std::setw(4) << npr ;
        if(npr == 1 ) std::cout << " pr(0): " << std::setw(2) << fl_msh->primitives[0] ;  
        std::cout << " xf:" << xform_string( fl_msh->xform )  ;
        std::cout << fl_msh->name << std::endl ; 
    }
}





void NScene::dump_node_transforms(const char* msg)
{
    LOG(info) << msg ; 

    typedef std::map<unsigned, std::array<float,16>> MIA ; 
    for(MIA::const_iterator it=m_xf.begin() ; it != m_xf.end() ; it++)
         std::cout 
              << " node_id " << std::setw(4) << it->first
              << std::endl 
              << " xf  "  << xform_string(it->second) 
              << std::endl 
              << " xf2 "  << xform_string(getNodeTransform(it->first))
              << std::endl 
              << " xf3 "  << xform_string(getFlatTransform(it->first))
              << std::endl ; 

}



/*

NScene::check_transforms stack.size 1
NScene::check_transforms node_id    0 nxfc    0
NScene::check_transforms node_id  524 nxfc    1
NScene::check_transforms node_id  523 nxfc    2
NScene::check_transforms node_id  522 nxfc    3
NScene::check_transforms node_id    1 nxfc    4
NScene::check_transforms node_id  521 nxfc    5
NScene::check_transforms node_id  520 nxfc    6
NScene::check_transforms node_id  519 nxfc    7
NScene::check_transforms node_id  518 nxfc    8
NScene::check_transforms node_id  517 nxfc    9
NScene::check_transforms node_id  516 nxfc   10
NScene::check_transforms node_id  515 nxfc   11

3157             if (node->mesh >= 0) {
3158 // BUG: initialization
3159 #ifdef _WIN32
3160                 auto fm = new fl_mesh();
3161                 fm->name = gltf->meshes.at(mesh_name).name;
3162                 fm->xform = xf;
3163                 fm->primitives = meshes.at(mesh_name);
3164                 fl_gltf->meshes.push_back(fm);
3165 #else
3166                 fl_gltf->meshes.push_back(
3167                     new fl_mesh{gltf->meshes.at(node->mesh).name, xf,
3168                         meshes.at(node->mesh)});
3169 #endif

     // order of fl_mesh vector is node traverse order...

3170                 fl_scn->meshes.push_back((int)fl_gltf->meshes.size() - 1);
3171             }
3172             for (auto child : node->children) { stack.push_back({child, xf}); }



*/





struct nd
{
   int idx ;
   int mesh ; 
   int depth ; 
   nd* parent ; 

   nmat4triple* transform ; 
   nmat4triple* gtransform ; 
   std::vector<nd*> children ; 

   std::string desc();
   static nmat4triple* make_global_transform(nd* n) ; 
};


std::string nd::desc()
{
    std::stringstream ss ; 

    ss << "nd"
       << " [" 
       << std::setw(3) << idx 
       << ":" 
       << std::setw(3) << mesh 
       << ":" 
       << std::setw(3) << children.size() 
       << ":" 
       << std::setw(2) << depth
       << "]"
       ;

    return ss.str();
}

nmat4triple* nd::make_global_transform(nd* n)
{
    std::vector<nmat4triple*> tvq ; 
    while(n)
    {
        if(n->transform) tvq.push_back(n->transform);
        n = n->parent ; 
    }
    return tvq.size() == 0 ? NULL : nmat4triple::product(tvq, true) ; 
}


void NScene::import()
{
    m_root = import_r(0, NULL, 0); 
    //dumpNdTree("NScene::import");
    compare_trees();
}

nd* NScene::import_r(int idx,  nd* parent, int depth)
{
    ygltf::node_t* node = getNode(idx);

    nd* n = new nd ; 

    n->idx = idx ; 
    n->mesh = node->mesh ; 
    n->parent = parent ;
    n->depth = depth ;
    n->transform = new nmat4triple( node->matrix.data() ); 
    n->gtransform = nd::make_global_transform(n) ;   

    for(auto child : node->children) n->children.push_back(import_r(child, n, depth+1));  // recursive call

    m_nd[idx] = n ;

    return n ; 
}

nd* NScene::getNd(int idx)
{
    return m_nd[idx] ; 
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

    assert( node->mesh == n->mesh );
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
            std::cout << "ntt " << xform_string( tt ) << std::endl ;    
            std::cout << "nmx " << xform_string( node->matrix ) << std::endl ;    
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
            std::cout << "gg  " << xform_string( gg ) << std::endl ;    
            std::cout << "fxf " << xform_string( fxf ) << std::endl ;    
            std::cout << "nxf " << xform_string( nxf ) << std::endl ;    
        }
        assert(global_match);
    }



    for(auto child : node->children) compare_trees_r( child );
}

