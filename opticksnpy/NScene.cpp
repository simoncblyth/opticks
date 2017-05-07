#include <sstream>
#include <iomanip>

#include "BFile.hh"

#include "NYGLTF.hpp"
#include "NScene.hpp"
#include "NCSG.hpp"

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
    load(m_base, m_name, m_scene_idx);
    collect_mesh_instances();
    collect_mesh_totals();
    load_mesh_extras();

    //check_transforms();
}




void NScene::load(const char* base, const char* name, int scene_idx )
{
    std::string path = BFile::FormPath(base, name);

    LOG(info) << "NScene::load"
              << " path " << path
              ;

    bool load_bin = true ; 
    bool load_shaders = true ; 
    bool load_img = false ; 
    bool skip_missing = true  ;   

    m_gltf = ygltf::load_gltf(path, load_bin, load_shaders, load_img, skip_missing ) ;
    m_fgltf = ygltf::flatten_gltf(m_gltf, scene_idx); 
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


/*
    for(auto mvi : m_mesh_instances)
    {
        auto mesh_id = mvi.first ; 
        auto instances = mvi.second ; 
        std::cout << " mesh_id " << mesh_id << " instances " << instances.size() << std::endl ; 
    }
*/
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

const std::array<float, 16>& NScene::getTransform(unsigned node_idx )
{
    assert(node_idx < m_fgltf->meshes.size() );   
    auto fl_msh = m_fgltf->meshes[node_idx] ;
    return fl_msh->xform ;
} 







std::string NScene::descNode( unsigned node_idx )
{
    assert(node_idx < m_fgltf->meshes.size() );   
    auto fl_msh = m_fgltf->meshes[node_idx] ;

    std::stringstream ss ; 
    ss
           << "[" << std::setw(4) << node_idx  << "]"
           << xform_string( fl_msh->xform )  
           << fl_msh->name 
           ;

    return ss.str() ;
}


 
std::string NScene::descInstance( unsigned mesh_idx, unsigned instance_idx )
{
    int node_idx = getInstanceNodeIndex(mesh_idx, instance_idx);
    std::stringstream ss ; 
    ss     
       << "[" << std::setw(3) << node_idx << ":" << std::setw(3) << instance_idx << "] "
       << descNode(node_idx )
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





void NScene::check_transforms(int scn_id)
{

    auto scn = &m_gltf->scenes.at(scn_id);

    // initialize stack of node transforms to identity matrix
    auto stack = std::vector<std::tuple<int, std::array<float, 16>>>();

    typedef std::map<int, std::array<float, 16>> MIA ; 
    MIA check  ;

    for (auto node_id : scn->nodes) 
    {
        stack.push_back(std::make_tuple(node_id, _identity_float4x4));
    }

    while (!stack.empty()) 
    {
        int              node_id;
        std::array<float, 16> xf;
        std::tie(node_id, xf)   = stack.back();
        stack.pop_back();   

        auto node = &m_gltf->nodes.at(node_id);
        check[node_id] = xf ;  // <-- this is failing to collect the expected transforms

        std::cout << "check_transforms"
                  << " node.id " << node_id 
                  << " node.mesh " << node->mesh 
                  << std::endl ;  


        xf = _float4x4_mul(xf, node_transform(node));   //   T-R-S-M    

        for (auto child : node->children) stack.push_back( {child, xf} ); 
    }



/*   
        if( node->mesh == 3)
        { 
            std::cout 
                << " node.id " << std::setw(3) << node_id 
                << " node.mesh " << std::setw(3) << node->mesh 
                << " node.name:" << node->name 
                << std::endl ; 
        
            std::cout << "lxf:" << xform_string(node->matrix) << std::endl ; 
            std::cout << "gxf: " << xform_string( xf ) << std::endl ; 
        }
*/



    LOG(info) << "NScene::check_transforms" 
              << " check.size " << check.size()
             ; 


/*
    for(MIA::const_iterator it=check.begin() ; it != check.end() ; it++)
    {
        int node_id = it->first ; 
        //auto oxf     = it->second ; 

        const std::array<float, 16>& nxf = getTransform(node_id) ;

        std::cout << "nxf: " << xform_string( nxf ) << std::endl ; 
        //std::cout << "oxf: " << xform_string( oxf ) << std::endl ; 

    }
*/




}










