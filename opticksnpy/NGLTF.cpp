#include "PLOG.hh"
#include "BFile.hh"

#include "NYGLTF.hpp"
#include "NGLTF.hpp"
#include "NPY.hpp"
#include "NGLMExt.hpp"


NGLTF::NGLTF(const char* base, const char* name, const char* config, unsigned scene_idx)
   :
    m_base(base ? strdup(base) : NULL),
    m_name(name ? strdup(name) : NULL),
    m_config(config ? strdup(config) : NULL),
    m_scene_idx(scene_idx),
    m_gltf(NULL),
    m_fgltf(NULL)
{
    load();
    collect();
}


const char* NGLTF::getConfig()
{
    return m_config ; 
}



void NGLTF::load()
{
    std::string path = BFile::FormPath(m_base, m_name);

    LOG(info) << "NGLTF::load"
              << " path " << path
              ;

    bool load_bin = true ; 
    bool load_shaders = true ; 
    bool load_img = false ; 
    bool skip_missing = true  ;   

    try
    {
        m_gltf = ygltf::load_gltf(path, load_bin, load_shaders, load_img, skip_missing ) ;
    }
    catch( ygltf::gltf_exception& e )
    {
        LOG(fatal) << "NGLTF::load FAILED FOR PATH " << path ; 
        assert(0);
    }
    catch(...)
    {
        LOG(fatal) << "NGLTF::load FAILED FOR PATH " << path ; 
        assert(0);
    }

    m_fgltf = ygltf::flatten_gltf(m_gltf, m_scene_idx); 


    LOG(info) << "NGLTF::load DONE"
              ;

}

void NGLTF::collect()
{
    auto scn = &m_gltf->scenes.at(m_scene_idx);

    auto stack = std::vector<std::tuple<int, std::array<float, 16>>>();

    for (auto node_id : scn->nodes)   // pushing root nodes, typically only one
    {
        stack.push_back(std::make_tuple(node_id, nglmext::_identity_float4x4));
    }

    assert( stack.size() == 1);

    while (!stack.empty()) // stack based recursion
    {
        int              node_id;
        std::array<float, 16> xf;
        std::tie(node_id, xf)   = stack.back();
        stack.pop_back();   

        auto node = &m_gltf->nodes.at(node_id);
        xf = nglmext::_float4x4_mul(xf, node_transform(node));   //   T-R-S-M    



        assert( m_xf.count(node_id) == 0 );
        assert( m_node2traversal.count(node_id) == 0 );

        m_node2traversal[node_id] = m_xf.size() ;  // <-- rosetta stone 
        m_xf[node_id] = xf ; 


        auto mesh_id = node->mesh ;
        if( m_mesh_totals.count(mesh_id) == 0) m_mesh_totals[mesh_id] = 0 ;  
        m_mesh_totals[mesh_id]++  ;
        m_mesh_used_globally[mesh_id] = false ; 

        if(m_mesh_instances.count(mesh_id) == 0) m_mesh_instances[mesh_id] = {} ;  
        m_mesh_instances[mesh_id].push_back(node_id)  ;


        for (auto child : node->children) stack.push_back( std::make_tuple(child, xf) ); 
    }
}




// meshes that are used globally need to have gtransform slots for all primitives
bool NGLTF::isUsedGlobally(unsigned mesh_idx)
{
    assert( m_mesh_used_globally.count(mesh_idx) == 1 );
    return m_mesh_used_globally[mesh_idx] ; 
}

void NGLTF::setIsUsedGlobally(unsigned mesh_idx, bool iug)
{
    m_mesh_used_globally[mesh_idx] = iug ; 
} 

ygltf::mesh_t* NGLTF::getMesh(unsigned mesh_id)
{
    return &m_gltf->meshes.at(mesh_id);
}


unsigned NGLTF::getNumMeshes()
{
    return m_mesh_instances.size() ;
}

unsigned NGLTF::getNumInstances(unsigned mesh_idx)
{
    assert( mesh_idx < m_mesh_instances.size() );
    return m_mesh_instances[mesh_idx].size() ; 
}

const std::vector<unsigned>& NGLTF::getInstances(unsigned mesh_idx)
{
    assert( mesh_idx < m_mesh_instances.size() );
    const std::vector<unsigned>& instances = m_mesh_instances[mesh_idx] ;
    return instances ; 
}
   
void NGLTF::dumpAllInstances( unsigned mesh_idx, unsigned maxdump )
{
    const std::vector<unsigned>& instances = getInstances(mesh_idx);
    unsigned ninst = std::min<unsigned>(instances.size(), maxdump) ; 
    for(unsigned i=0 ; i < ninst ; i++) std::cout << descNode(instances[i]) << std::endl ; 
    if(ninst < instances.size()) std::cout << "..." << std::endl ; 
}

void NGLTF::dumpAll()
{
    unsigned num_meshes = getNumMeshes() ;
    for(unsigned i=0 ; i < num_meshes ; i++) dumpAllInstances(i) ; 
}


NPY<float>* NGLTF::makeInstanceTransformsBuffer(unsigned mesh_idx)
{
    const std::vector<unsigned>& instances = getInstances(mesh_idx);

    NPY<float>* buf = NPY<float>::make(instances.size(), 4, 4);
    buf->zero();

    for(unsigned i=0 ; i < instances.size() ; i++)
    {
        int node_idx = instances[i] ; 
        glm::mat4 xf = getTransformMatrix(node_idx) ;
        buf->setMat4(xf, i);
    }

    if(mesh_idx == 0) 
    {
       assert(buf->getNumItems() == 1);
    }
    return buf ;
}



void NGLTF::dump_mesh_totals(const char* msg, unsigned maxdump)
{
    LOG(info) << msg ; 

    int node_total = 0 ; 

    typedef std::map<unsigned, unsigned> UU ; 
    for(UU::const_iterator it=m_mesh_totals.begin() ; it != m_mesh_totals.end() ; it++)
    {
       unsigned mesh_idx = it->first ;  
       unsigned mesh_count = it->second ;  
       unsigned num_instances = getNumInstances(mesh_idx);
       assert(num_instances == mesh_count ); 
       const std::vector<unsigned>& instances = getInstances(mesh_idx);

       auto mesh = &m_gltf->meshes.at(mesh_idx);
       node_total += mesh_count ; 
       std::cout 
             << std::setw(4) << mesh_idx  
             << " : " 
             << std::setw(4) << mesh_count
             << " : "
             << std::setw(60) << mesh->name
             << " : "
             ;

       unsigned ninst = std::min<unsigned>(instances.size(), maxdump) ; 

       std::cout << " ( " ; 
       for(unsigned i=0 ; i < ninst ; i++)  std::cout << " " << std::setw(4) << instances[i] ;
       std::cout << ( ninst > 1 && ninst < instances.size() ? " ..." : "" ) << " ) " ; 
       std::cout << std::endl ; 

    }
    std::cout << " node_total " << node_total << std::endl ;  
}




ygltf::fl_mesh* NGLTF::getFlatNode(unsigned node_idx)
{
    assert( m_node2traversal.count(node_idx) == 1 );
    unsigned traversal_idx = m_node2traversal[node_idx] ;  
    assert( traversal_idx < m_fgltf->meshes.size() );   
    // must use this rosetta stone as  m_fgltf->meshes in traversal order, not node_idx order
    ygltf::fl_mesh* fln = m_fgltf->meshes[traversal_idx] ;  
    return fln ; 
}

const std::array<float, 16>& NGLTF::getFlatTransform(unsigned node_idx )
{
    ygltf::fl_mesh* fln = getFlatNode(node_idx);
    return fln->xform ;
} 

const std::array<float,16>& NGLTF::getNodeTransform(unsigned node_idx)
{
    // yoctogl doesnt make it easy to look up the transform for a node
    // so collected m_xf in collect_node_transforms
    return m_xf[node_idx] ; 
}

glm::mat4 NGLTF::getTransformMatrix( unsigned node_idx )
{
    const std::array<float, 16>& nxf = getNodeTransform(node_idx);
    const std::array<float, 16>& fxf = getFlatTransform(node_idx);
    assert( nxf == fxf );
    return glm::make_mat4( nxf.data() );
}



unsigned  NGLTF::getNumNodes() const 
{
   return m_gltf->nodes.size()  ;
}

ygltf::node_t* NGLTF::getNode(unsigned node_idx) const 
{
    assert(node_idx < m_gltf->nodes.size() );   
    return &m_gltf->nodes.at(node_idx);
}  

std::string NGLTF::descNode( unsigned node_idx )
{
    const std::array<float, 16>& nxf = getNodeTransform(node_idx);
    ygltf::node_t* node = getNode(node_idx);
    std::stringstream ss ; 
    ss
           << "[" << std::setw(4) << node_idx  << "]"
           << " node.mesh " << std::setw(3) << node->mesh 
           << " node.nchild " << std::setw(3) << node->children.size() 
           << " node.xf " << nglmext::xform_string( nxf )  
        //   << " node.name " << node->name 
           ;
    return ss.str() ;
}

std::string NGLTF::descFlatNode( unsigned node_idx )
{
    ygltf::fl_mesh* fln = getFlatNode(node_idx);
    std::stringstream ss ; 
    ss
           << "[" << std::setw(4) << node_idx  << "]"
           << nglmext::xform_string( fln->xform )  
           << fln->name 
           ;
    return ss.str() ;
}



void NGLTF::dump(const char* msg)
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

void NGLTF::dump_scenes(const char* msg)
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

void NGLTF::dump_flat_nodes(const char* msg)
{
    LOG(info) << msg << " " << m_fgltf->meshes.size() ; 

    for(auto fl_msh : m_fgltf->meshes)
    {
        auto npr = fl_msh->primitives.size() ;
        std::cout << "pr:" << std::setw(4) << npr ;
        if(npr == 1 ) std::cout << " pr(0): " << std::setw(2) << fl_msh->primitives[0] ;  
        std::cout << " xf:" << nglmext::xform_string( fl_msh->xform )  ;
        std::cout << fl_msh->name << std::endl ; 
    }
}





void NGLTF::dump_node_transforms(const char* msg)
{
    LOG(info) << msg ; 

    typedef std::map<unsigned, std::array<float,16>> MIA ; 
    for(MIA::const_iterator it=m_xf.begin() ; it != m_xf.end() ; it++)
         std::cout 
              << " node_id " << std::setw(4) << it->first
              << std::endl 
              << " xf  "  << nglmext::xform_string(it->second) 
              << std::endl 
              << " xf2 "  << nglmext::xform_string(getNodeTransform(it->first))
              << std::endl 
              << " xf3 "  << nglmext::xform_string(getFlatTransform(it->first))
              << std::endl ; 

}


