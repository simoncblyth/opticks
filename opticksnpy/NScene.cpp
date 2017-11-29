#include <sstream>
#include <iomanip>

#include "SSys.hh"

#include "BStr.hh"
#include "BFile.hh"


#include "GLMFormat.hpp"
#include "NGLMExt.hpp"
#include "NYGLTF.hpp"

#include "Counts.hpp"
#include "NTrianglesNPY.hpp"
#include "NParameters.hpp"
#include "NPY.hpp"
#include "NNode.hpp"
#include "NBBox.hpp"
#include "NSceneConfig.hpp"


#include "NSDF.hpp"
#include "NTxt.hpp"
#include "NCSG.hpp"
#include "NGLMExt.hpp"
#include "NGLMCF.hpp"
#include "Nd.hpp"

#include "N.hpp"
#include "NScene.hpp"

#include "PLOG.hh"



NSceneConfig* NScene::getConfig()
{
   return m_config ;  
}

unsigned NScene::getNumNd() const 
{
   return m_nd.size();
}

nd* NScene::getNd(unsigned idx) const 
{
    std::map<unsigned, nd*>::const_iterator it = m_nd.find(idx) ; 
    return it == m_nd.end() ? NULL : it->second  ; 
    // operator[] cannot be const 
}
nd* NScene::getRoot() const 
{
    return m_root ; 
}
NCSG* NScene::getCSG(unsigned mesh_idx) const 
{
    std::map<unsigned,NCSG*>::const_iterator it = m_csg.find(mesh_idx) ; 
    return it == m_csg.end() ? NULL : it->second  ;
}



NCSG* NScene::findCSG(const char* q_soname, bool startswith) const 
{
    typedef std::map<unsigned,NCSG*>::const_iterator MUCI ;

    NCSG* q_csg = NULL ; 

    for(MUCI it=m_csg.begin() ; it != m_csg.end() ; it++)
    {
        unsigned mesh_idx = it->first ;
        std::string _soname = soname(mesh_idx);

        NCSG* csg = it->second ; 

        bool pick = startswith ? BStr::StartsWith(_soname.c_str(), q_soname ) : strcmp(_soname.c_str(), q_soname ) == 0 ;    
        if(pick)
        {
            q_csg = csg ; 
            break ; 
        }
    }
    return q_csg ; 
} 

void NScene::collect_mesh_nodes(std::vector<unsigned>& nodes, unsigned mesh) const 
{
    collect_mesh_nodes_r(m_root, nodes, mesh);
}
void NScene::collect_mesh_nodes_r(nd* n, std::vector<unsigned>& nodes, unsigned mesh) const 
{
    if(n->mesh == mesh) nodes.push_back(n->idx);
    for(nd* c : n->children) collect_mesh_nodes_r(c, nodes, mesh);
}

std::string NScene::present_mesh_nodes(std::vector<unsigned>& nodes, unsigned dmax) const 
{
     std::stringstream ss ; 
     ss << " nds[" << std::setw(3) << nodes.size() << "] " ;  
     for(unsigned i=0 ; i < std::min<unsigned>(nodes.size(), dmax) ; i++) ss << " " << nodes[i] ; 
     ss << ( nodes.size() > dmax ? " ... " : " . " ) ;     
     return ss.str();
}
    


bool NScene::Exists(const char* base, const char* name)
{
    return BFile::ExistsFile(base, name);
}

long NScene::SecondsSinceLastWrite(const char* base, const char* name)
{
    std::time_t* slwt = BFile::SinceLastWriteTime(base, name);
    long age = slwt ? *slwt : -1 ; 
    return age ;  
}


NScene* NScene::Load( const char* gltfbase, const char* gltfname, const char* idfold, NSceneConfig* gltfconfig, int dbgnode) 
{
    NScene* scene =  NScene::Exists(gltfbase, gltfname) ? new NScene(gltfbase, gltfname, idfold, gltfconfig, dbgnode) : NULL ;

    if(!scene)
        LOG(fatal) << "NScene:Load MISSING PATH" 
                   << " gltfbase " << gltfbase
                   << " gltfname " << gltfname
                   << " gltfconfig " << gltfconfig
                   ; 


    return scene ; 
}


// from gltfconfig
NSceneConfigBBoxType NScene::bbox_type() const 
{
    return m_config->bbox_type();
}
const char* NScene::bbox_type_string() const 
{
    return m_config->bbox_type_string();
}




NScene::NScene(const char* base, const char* name, const char* idfold, NSceneConfig* config, int dbgnode, int scene_idx)  
   :
    NGLTF(base, name, config, scene_idx),
    m_num_gltf_nodes(getNumNodes()),
    m_idfold(idfold ? strdup(idfold) : NULL),
    m_config(config),
    m_dbgnode(dbgnode),
    m_containment_err(0),
    m_verbosity(m_config->verbosity),
    m_num_global(0),
    m_num_csgskip(0),
    m_num_placeholder(0),
    m_num_selected(0),
    m_csgskip_lvlist(NULL),
    m_placeholder_lvlist(NULL),
    m_node_count(0),
    m_label_count(0),
    m_digest_count(new Counts<unsigned>("progenyDigest")),
    m_age(NScene::SecondsSinceLastWrite(base, name)),
    m_triple_debug(true),
    m_triple(NULL),
    m_num_triple(0)
{
    init_lvlists();
    init();
}

void NScene::init()
{
    load_asset_extras();  // includes verbosity from glTF 

    if(m_verbosity > 0)
    LOG(info) << "NScene::init START"
              << " age(s) " << m_age 
              << " days " << std::fixed << std::setw(7) << std::setprecision(3) << float(m_age)/float(60*60*24) 
              << " num_gltf_nodes " << m_num_gltf_nodes
              ;  

    if(m_triple_debug)
    { 
        m_triple = NPY<float>::make( m_num_gltf_nodes, 3, 4, 4 ); // debug TVQ collection
        m_triple->zero();
    }

    load_csg_metadata();

    LOG(info) << "NScene::init import_r START " ; 

    m_root = import_r(0, NULL, 0); 

    LOG(info) << "NScene::init import_r DONE " ; 

    if(m_triple_debug)
    {
        LOG(info) << "NScene::init triple_debug "
                  << " num_gltf_nodes " << m_num_gltf_nodes
                  << " triple_mismatch " << m_num_triple_mismatch
                  ;
        m_triple->save("$TMP/NScene_triple.npy");
    }
 


    postimportnd();

    if(m_verbosity > 1)
    dumpNdTree("NScene::NScene");

    compare_trees();


    count_progeny_digests();

    find_repeat_candidates();

    dump_repeat_candidates();

    if(m_config->disable_instancing > 0)
    { 
        LOG(warning) << "NScene::init disable_instancing via gltfconfig " ; 
    }
    else
    { 
        labelTree();
    } 
    

    //if(m_verbosity > 1)
    dumpRepeatCount(); 

    markGloballyUsedMeshes_r(m_root);

    // move load_mesh_extras later so can know which meshes are non-instanced needing 
    // gtransform slots for all primitives
    load_mesh_extras();

    postimportmesh();

    write_lvlists();

    if(m_verbosity > 0)
    LOG(info) << "NScene::init DONE" ;  
    //assert(0 && "hari kari");
}




void NScene::init_lvlists()
{
    if(!m_idfold)
    {
         LOG(info) << "NScene::init_lvlists"
                   << " lvlist writing requires idfold " ;
         return ; 
    }

  
    std::string stem = BFile::Stem(m_name);  // eg "g4_00" 
    std::string csgskip_path = BFile::FormPath(m_idfold, stem.c_str(), "CSGSKIP_DEEP_TREES.txt");
    std::string placeholder_path = BFile::FormPath(m_idfold, stem.c_str(), "PLACEHOLDER_FAILED_POLY.txt");

    LOG(info) << " csgskip_path " << csgskip_path ; 
    LOG(info) << " placeholder_path " << placeholder_path ; 


    BFile::preparePath( csgskip_path.c_str(), true);
    BFile::preparePath( placeholder_path.c_str(), true);

    m_csgskip_lvlist     = new NTxt(csgskip_path.c_str());
    m_placeholder_lvlist = new NTxt(placeholder_path.c_str());


    if(m_verbosity > 2)
    {
        LOG(info) << "NScene::init_lvlists" ;
        std::cout << " csgskip " << m_csgskip_lvlist->desc() << std::endl ; 
        std::cout << " placeholder(polyfail) " << m_placeholder_lvlist->desc() << std::endl ; 
    }

}

void NScene::write_lvlists()
{
    if(!m_idfold)
    {
         LOG(info) << "NScene::write_lvlists"
                   << " lvlist writing requires idfold " ;
         return ; 
    }

    if(m_verbosity > 2)
    {
        LOG(info) << "NScene::write_lvlists";
        std::cout << " csgskip " << m_csgskip_lvlist->desc() << std::endl ; 
        std::cout << " placeholder(polyfail) " << m_placeholder_lvlist->desc() << std::endl ; 
    }

    m_csgskip_lvlist->write();
    m_placeholder_lvlist->write();
}


void NScene::load_asset_extras()
{
    auto extras = m_gltf->asset.extras ; 
    unsigned extras_verbosity = extras["verbosity"]; 

    if(extras_verbosity > m_verbosity)
    {
        LOG(warning) << "NScene::load_asset_extras"
                     << " verbosity increase from scene gltf "
                     << " extras_verbosity " << extras_verbosity
                     << " m_verbosity " << m_verbosity
                     ;

        m_verbosity = extras_verbosity ; 
       
    }

    m_targetnode = extras["targetnode"]; 

    if(m_verbosity > 1)
    LOG(info) << "NScene::load_asset_extras"
              << " m_verbosity " << m_verbosity 
              << " m_targetnode " << m_targetnode 
               ;
}

unsigned NScene::getVerbosity()
{
    return m_verbosity ; 
}
unsigned NScene::getTargetNode()
{
    return m_targetnode ; 
}



void NScene::load_csg_metadata()
{
   // for debugging need the CSG metadata prior to loading the trees
    unsigned num_meshes = getNumMeshes();
    if(m_verbosity > 0)
    LOG(info) << "NScene::load_csg_metadata"
              << " verbosity " << m_verbosity 
              << " num_meshes " << num_meshes
              ;

    for(std::size_t mesh_id = 0; mesh_id < num_meshes; ++mesh_id)
    {
        ygltf::mesh_t* mesh = getMesh(mesh_id);
        auto extras = mesh->extras ; 

        std::string uri = extras["uri"] ; 

        std::string soName = extras["soName"] ; 

        int lvIdx = extras["lvIdx"] ; 
        m_csg_lvIdx[mesh_id] = lvIdx ; 

        std::string csgpath = BFile::FormPath(m_base, uri.c_str() );

        NParameters* meta = NCSG::LoadMetadata(csgpath.c_str());
        m_csg_metadata[mesh_id] = meta ; 

        std::string meta_soname = soname(mesh_id);
        assert( meta_soname.compare(soName) == 0) ; 

        if(m_verbosity > 3)
        LOG(info) << "NScene::load_csg_metadata"
                  << " verbosity " << m_verbosity 
                  << " mesh_id " << std::setw(3) << mesh_id
                  << " lvIdx " << std::setw(6) << lvIdx
                  << " soName " << soName
                  ;


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

int          NScene::lvidx(unsigned mesh_id) const { return m_csg_lvIdx.at(mesh_id) ; }

// keys need to match analytic/sc.py 
std::string NScene::lvname(unsigned mesh_id) const { return getCSGMeta<std::string>(mesh_id,"lvname","-") ; }
std::string NScene::soname(unsigned mesh_id) const { return getCSGMeta<std::string>(mesh_id,"soname","-") ; }
int         NScene::height(unsigned mesh_id) const { return getCSGMeta<int>(mesh_id,"height","-1") ; }
int         NScene::nchild(unsigned mesh_id) const { return getCSGMeta<int>(mesh_id,"nchild","-1") ; }
bool        NScene::isSkip(unsigned mesh_id) const { return getCSGMeta<int>(mesh_id,"skip","0") == 1 ; }

std::string NScene::meshmeta(unsigned mesh_id) const 
{
    std::stringstream ss ; 
    ss 
       << "NScene::meshmeta"
       << " mesh_id " << std::setw(3) << mesh_id
       << " lvidx "   << std::setw(3) << lvidx(mesh_id)
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

    if(m_verbosity > 1)
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

        int lvidx_ = lvidx(mesh_id);

        NCSG* csg = NCSG::LoadTree(csgpath.c_str(), m_config ); 
        csg->setIndex(mesh_id);


        bool csgskip = csg->isSkip() ;
        if(csgskip) 
        {
            if(m_csgskip_lvlist)
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
            if(m_placeholder_lvlist)
                m_placeholder_lvlist->addLine(mesh->name);
            m_num_placeholder++ ; 
        }


        m_csg[mesh_id] = csg ; 
    

        if(m_verbosity > 1)
        std::cout << " mId " << std::setw(4) << mesh_id 
                  << " lvidx " << std::setw(4) << lvidx_
                  << " npr " << std::setw(4) << primitives.size() 
                  << " nam " << std::setw(65) << mesh->name 
                  << " iug " << std::setw(1) << iug 
                  << " poly " << std::setw(3) << tris->getPoly()
                  << " smry " << csg->smry() 
                  << std::endl ; 
    }  


    if(m_verbosity > 1)
    LOG(info) << "NScene::load_mesh_extras DONE"
              << " m_verbosity " << m_verbosity
              << " num_meshes " << num_meshes
              << " m_num_global " << m_num_global
              << " m_num_csgskip " << m_num_csgskip
              << " m_num_placeholder " << m_num_placeholder
              ;

}


void NScene::dumpCSG(const char* dbgmesh, const char* msg) const 
{
    unsigned num_csg = m_csg.size() ;
    LOG(info) << msg 
              << " num_csg " << num_csg
              << " dbgmesh " << ( dbgmesh ? dbgmesh : "-" )
              ;

    if(dbgmesh == NULL)
    {
        for(unsigned i=0 ; i < num_csg ; i++)
        {
            NCSG* csg = getCSG(i) ;
            assert(csg);
            std::cout << std::setw(4) << i 
                      << csg->brief()
                      << std::endl ; 
        }
    }
    else
    { 
        bool startswith = true ; 
        NCSG* csg = findCSG(dbgmesh, startswith) ;
        if(csg)
        {
            csg->dump();
            csg->dump_surface_points("dsp", 200);

            unsigned mesh_id = csg->getIndex();

            std::vector<unsigned> nodes ; 
            collect_mesh_nodes(nodes, mesh_id);
            std::cout << present_mesh_nodes(nodes, 20) << std::endl ; 

        }
        else
        {
            LOG(warning) << "failed to findCSG with soname " << dbgmesh ; 
        } 
    }
}



nmat4triple* NScene::make_triple( const float* data)
{
    // spell out nglmext::invert_trs for debugging discrepancies

    glm::mat4 T = glm::make_mat4(data) ;
    ndeco d = nglmext::polar_decomposition( T ) ;

    glm::mat4 isirit = d.isirit ; 
    glm::mat4 i_trs = glm::inverse( T ) ; 

    NGLMCF cf(isirit, i_trs );

    if(!cf.match) 
    {
        m_num_triple_mismatch++ ; 
        //LOG(warning) << cf.desc("NScene::make_triple polar_decomposition inverse and straight inverse are mismatched " );
    }

    glm::mat4 V = isirit ;
    glm::mat4 Q = glm::transpose(V) ;

    nmat4triple* tvq = new nmat4triple(T, V, Q); 

    if(m_triple)  // collecting triples for mismatch debugging 
    {
        m_triple->setMat4Triple( tvq , m_num_triple++ );
    }
    return tvq ; 
}

nd* NScene::import_r(int idx,  nd* parent, int depth)
{
    ygltf::node_t* ynode = getNode(idx);
    auto extras = ynode->extras ; 
    std::string boundary = extras["boundary"] ; 
    std::string pvname = extras["pvname"] ; 
    unsigned selected = extras["selected"] ; 
 
    nd* n = new nd ;   // NB these are structural nodes, not CSG tree nodes

    n->idx = idx ; 
    n->repeatIdx = 0 ; 
    n->mesh = ynode->mesh ; 
    n->parent = parent ;
    n->depth = depth ;
    n->boundary = boundary ;
    n->pvname = pvname ; 
    n->selected = selected ;  // TODO: get rid of this, are now doing selection in GScene 
    n->containment = 0 ; 

    n->transform = make_triple(ynode->matrix.data()) ; 
    n->gtransform = nd::make_global_transform(n) ;   

    if(selected) m_num_selected++ ; 
 

    for(int child : ynode->children) n->children.push_back(import_r(child, n, depth+1));  // recursive call

    m_nd[idx] = n ;

    return n ; 
}



void NScene::postimportnd()
{



    const nd* dn = m_dbgnode > -1 ? getNd(m_dbgnode) : NULL ;
    if( dn )
    {
        m_dbgnode_list.push_back(dn->idx);
        //if(dn->parent) m_dbgnode_list.push_back(dn->parent->idx);
    }

    LOG(info) << "NScene::postimportnd" 
              << " numNd " << getNumNd()
              << " num_selected " << m_num_selected
              << " dbgnode " << m_dbgnode
              << " dbgnode_list " << m_dbgnode_list.size()
              << " verbosity " << m_verbosity
               ; 
}


bool NScene::is_dbgnode( const nd* n) const 
{ 
    return std::find(m_dbgnode_list.begin(), m_dbgnode_list.end(), n->idx ) != m_dbgnode_list.end() ;
}


void NScene::postimportmesh()
{
    LOG(info) << "NScene::postimportmesh" 
              << " numNd " << getNumNd()
              << " dbgnode " << m_dbgnode
              << " dbgnode_list " << m_dbgnode_list.size()
              << " verbosity " << m_verbosity
               ; 

    m_config->dump("NScene::postimportmesh.cfg");

    if(m_config->check_aabb_containment > 0)
    check_aabb_containment();

    if(m_config->check_surf_containment > 0)
    check_surf_containment();


    if(SSys::IsHARIKARI())
    assert( 0 && "NScene::postimportmesh HARIKARI");
}

//  tgltf-;HARIKARI=1 tgltf-t  




nnode* NScene::getSolidRoot(const nd* n) const 
{
    // provides mesh level CSG root nnode associated to the structural nd 
    // NB this will be the same for all instances of that mesh within the nd structure

    unsigned mesh_idx = n->mesh ; 
    NCSG* csg = getCSG(mesh_idx);
    assert(csg);
    nnode* root = csg->getRoot();
    assert(root);
    return root ; 
}


nbbox NScene::calc_aabb(const nd* n, bool global) const 
{
    assert( n->gtransform );
    const glm::mat4& nt  = n->gtransform->t ; 

    const nnode* solid = getSolidRoot(n);
    assert(solid);

    nbbox bb  = solid->bbox();
    nbbox gbb = bb.make_transformed(nt) ; 

    if(m_verbosity > 2)
    std::cout 
        << " get_bbox "
        << " verbosity " << m_verbosity 
        << " n.mesh "  << n->mesh
        << " solid.tag "  << solid->tag() 
        << std::endl 
        << gpresent("n.t", nt)
        << std::endl 
        << " bb  " <<  bb.desc() << std::endl 
        << " gbb " <<  gbb.desc() << std::endl 
        ;

    return global ? gbb : bb ; 
}

void NScene::update_aabb() 
{
    update_aabb_r(m_root); 
}
void NScene::update_aabb_r(nd* n) 
{
    bool global = true ; 
    n->aabb = calc_aabb( n, global ) ;   // probably makes more sense for this to go in NCSG, from whence transform as need ?
    for(nd* c : n->children) update_aabb_r(c) ;
}



void NScene::check_surf_containment() 
{
    LOG(info) << "NScene::check_surf_containment (csc)"
              << " verbosity " << m_verbosity 
              ;

    check_surf_containment_r(m_root);

    unsigned tot = getNumNd() ;

    LOG(info) << "NScene::check_surf_containment (csc)"
              << " verbosity " << m_verbosity 
              << " tot " << tot
              << " surferr " << gpresent(m_surferr)
              ; 
}
void NScene::check_surf_containment_r(const nd* n) 
{
    glm::uvec4 err = check_surf_points(n);

    if(err.x > 0) m_surferr.x++ ; 
    if(err.y > 0) m_surferr.y++ ; 
    if(err.z > 0) m_surferr.z++ ; 
    if(err.w > 0) m_surferr.w++ ; 

    if(is_dbgnode(n)) debug_node(n);

    for(const nd* c : n->children) check_surf_containment_r(c) ;
}





float NScene::sdf_procedural( const nd* n, const glm::vec3& q_) const 
{
    assert(0 && "dont use this use NSDF ");
    // distance from global frame query point to the surface of the solid associated to the structural nd

    glm::vec4 q(q_,1.0); 

    if(n->gtransform) q = n->gtransform->v * q ;  // apply inverse transform to take global position into frame of the structural nd 

    const nnode* solid = getSolidRoot(n);

    return (*solid)(q.x, q.y, q.z);
}



glm::uvec4 NScene::check_surf_points( const nd* n ) const // Classifiying the surface points of a node against the SDF of its parent 
{
    // intended to provide summary checking of all nodes, 
    // for more verbose indivdual node dumping use NScene::debug_node

    glm::uvec4 err(0,0,0,0) ; 
    bool dbgnode = is_dbgnode(n) ;

    const nd* p = n->parent ? n->parent : n ;  // only root has no parent

    NCSG* ncsg = getCSG(n->mesh);
    NCSG* pcsg = getCSG(p->mesh);

    int nlvid = lvidx(n->mesh);

    nnode* nroot = ncsg->getRoot();
    nnode* proot = pcsg->getRoot();

    const nmat4triple* id = nmat4triple::make_identity() ;
    N pp(proot, id);           
    N nn(nroot, n->transform );


    // checking sdf against of own local points is mainly a machinery test
    // but also shows the kind of precision are getting 

    bool self_check = false ;
    if(self_check)
    {
        pp.classify( pp.local, 1e-3, POINT_SURFACE );
        nn.classify( nn.local, 1e-3, POINT_SURFACE );
        err.z = pp.nsdf.tot.w ; 
        err.w = nn.nsdf.tot.w ; 

        std::cout << "NSc::csp" << " n " << std::setw(5) << n->idx << " p " << std::setw(5) << p->idx << " n.pv " << n->pvtag() << ( dbgnode ? " DEBUG_NODE " : " " ) << std::endl ; 
        std::cout << "pp.classify(pp.local) " << pp.desc() << std::endl ; 
        std::cout << "nn.classify(nn.local) " << nn.desc() << std::endl ; 
    }

    // Cross checking containment of a nodes points inside its parent 
    // OR vice versa checking that parents points are outside the child node
    // is the raison d'etre of this method.
    //
    // Coincidence is a problem, as well as impingement ... but try to 
    // see how big the issue is

    pp.classify( nn.local, 1e-3, POINT_INSIDE );

    //nn.classify( pp.local, 1e-3, POINT_OUTSIDE );
    err.x = pp.nsdf.tot.w ; 
    //err.y = nn.nsdf.tot.w ; 

    {
        std::cout << "NSc::csp" 
                  << " n " << std::setw(5) << n->idx 
                  << " nlv " << std::setw(3) << nlvid 
                  << " p " << std::setw(5) << p->idx 
                  << " n.pv " << std::setw(30) << n->pvtag() 
                  << " pp(nn.local) " << pp.desc()
                  << ( dbgnode ? " DEBUG_NODE " : " " )
                  ;

        //std::cout << "nn(pp.local) " << nn.desc() ;
        std::cout << std::endl  ;
    }

    return err ;  

}



void NScene::debug_node(const nd* n) const 
{
    //  DBGNODE=3159 NSceneLoadTest 

    bool dbgnode = is_dbgnode(n) ;
    if(!dbgnode) return ; 

    const nd* p = n->parent ? n->parent : n ;  // only root has no parent
    int nlvid = lvidx(n->mesh);
    int plvid = lvidx(p->mesh);

    LOG(info) << "NScene::debug_node " 
              << " n " << std::setw(5) << n->idx 
              << " n.mesh " << std::setw(5) << n->mesh
              << " n.lv " << std::setw(3) << nlvid 
              << " p.lv " << std::setw(3) << plvid 
              << " p " << std::setw(5) << p->idx 
              << " n.pv " << std::setw(30) << n->pvtag()
              ; 
 
    const_cast<nd*>(n)->dump_transforms("n->dump_transforms" ); 

    NCSG* ncsg = getCSG(n->mesh);
    NCSG* pcsg = getCSG(p->mesh);


    nnode* nroot = ncsg->getRoot();
    nnode* proot = pcsg->getRoot();

    const nmat4triple* id = nmat4triple::make_identity() ;
    N pp(proot, id);           
    N nn(nroot, n->transform );

    nn.dump_points("nn.dump_points");


    bool dump = true ; 
    glm::uvec4 err(0,0,0,0);

    LOG(info) << "pp.classify(pp.local)" ;
    pp.classify( pp.local, 1e-3, POINT_SURFACE, dump );
    err.z = pp.nsdf.tot.w ; 
    std::cout << "pp.classify(pp.local) " << pp.desc() << std::endl ; 

    LOG(info) << "nn.classify(nn.local)" ;
    nn.classify( nn.local, 1e-3, POINT_SURFACE, dump );
    err.w = nn.nsdf.tot.w ; 
    std::cout << "nn.classify(nn.local) " << nn.desc() << std::endl ; 

    LOG(info) << "pp.classify(nn.local)" ;
    pp.classify( nn.local, 1e-3, POINT_INSIDE,  dump );
    err.x = pp.nsdf.tot.w ; 
    std::cout << "pp(nn.local) " << pp.desc() << std::endl   ;

    LOG(info) << "nn.classify(pp.local)" ;
    nn.classify( pp.local, 1e-3, POINT_OUTSIDE, dump );
    err.y = nn.nsdf.tot.w ; 
    std::cout << "nn(pp.local) " << nn.desc() << std::endl   ;

}










void NScene::check_aabb_containment() 
{
    LOG(info) << "NScene::check_aabb_containment (cac)"
              << " verbosity " << m_verbosity 
              ;
    update_aabb();
    check_aabb_containment_r(m_root);

    unsigned tot = getNumNd() ;

    LOG(info) << "NScene::check_aabb_containment (cac)"
              << " verbosity " << m_verbosity 
              << " tot " << tot
              << " err " << m_containment_err 
              << " err/tot " << std::setw(10) << std::fixed << std::setprecision(2) << float(m_containment_err)/float(tot)
              ; 
}


void NScene::check_aabb_containment_r(const nd* n) 
{
    const nd* p = n->parent ? n->parent : n ;  // only root should not have parent

    const nbbox& nbb = n->aabb ;
    const nbbox& pbb = p->aabb ; 

    float epsilon = 1e-5 ; 

    unsigned errmask = nbb.classify_containment( pbb, epsilon );

    //n->containment = errmask ;  

    if(errmask) m_containment_err++ ; 

    //if(m_verbosity > 2 || ( errmask && m_verbosity > 0))
    {
        glm::vec3 dmin( nbb.min.x - pbb.min.x, nbb.min.y - pbb.min.y, nbb.min.z - pbb.min.z ); 
        glm::vec3 dmax( pbb.max.x - nbb.max.x, pbb.max.y - nbb.max.y, pbb.max.z - nbb.max.z ); 
        std::cout 
             << "NSc::cac"
             << " n " << std::setw(6) << n->idx
             << " p " << std::setw(6) << p->idx 
             << " mn(n-p) " << gpresent( dmin ) 
             << " mx(p-n) " << gpresent( dmax ) 
             << " n.pv " << std::setw(30) <<  n->pvtag()
             << " err " << nbbox::containment_mask_string( errmask ) 
             << std::endl 
             ;

        if(m_verbosity > 3 || ( errmask && m_verbosity > 1))
        std::cout 
             << " nbb " <<  nbb.desc() << std::endl 
             << " pbb " <<  pbb.desc() << std::endl 
             ;

    }

    for(const nd* c : n->children) check_aabb_containment_r(c) ;
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

    //if(m_verbosity > 1)
    LOG(info) << "NScene::count_progeny_digests"
              << " verbosity " << m_verbosity 
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




struct NRepeat
{
    unsigned   repeat_min ;  
    unsigned   vertex_min ;  
    unsigned    index ; 
    std::string pdig ; 
    unsigned    num_pdig ; 
    nd*         first ;   // cannot const as collection is deferred
    unsigned    num_progeny ; 
    bool        candidate ; 
    bool        select ; 
    

    bool isListed(const std::vector<std::string>& pdigs_)
    {   
        return std::find(pdigs_.begin(), pdigs_.end(), pdig ) != pdigs_.end() ;   
    }   

    NRepeat( unsigned repeat_min_, unsigned vertex_min_,  unsigned index_, const std::string& pdig_, unsigned num_pdig_, nd* first_ ) 
          :   
          repeat_min(repeat_min_),
          vertex_min(vertex_min_),
          index(index_),
          pdig(pdig_), 
          num_pdig(num_pdig_), 
          first(first_),
          num_progeny(first->get_progeny_count()),
          candidate(num_pdig > repeat_min),
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
              << " num_pdig "  << std::setw(6) << num_pdig
              << " num_progeny " <<  std::setw(6) << num_progeny 
             ;
        return ss.str();
    }
};





void NScene::find_repeat_candidates()
{
   // hmm : this approach will not gang together siblings 
   //       that always appear together, only subtrees 

    unsigned repeat_min = m_config->instance_repeat_min ; 
    unsigned vertex_min = m_config->instance_vertex_min ;  // hmm but dont have access to vertex counts at this stage 

    unsigned int num_progeny_digests = m_digest_count->size() ;

    //if(m_verbosity > 1)
    LOG(info) << "NScene::find_repeat_candidates"
               << " verbosity " << m_verbosity 
               << " config.instance_repeat_min " << repeat_min
               << " config.instance_vertex_min " << vertex_min
               << " num_progeny_digests " << num_progeny_digests 
              ;   

    std::vector<NRepeat> cands ; 

    for(unsigned i=0 ; i < num_progeny_digests ; i++)
    {   
        std::pair<std::string,unsigned int>&  kv = m_digest_count->get(i) ;

        std::string& pdig = kv.first ; 
        unsigned int num_pdig = kv.second ;   
        nd* first = m_root->find_node(pdig) ;

        NRepeat cand(repeat_min,vertex_min, i, pdig, num_pdig , first );
        cands.push_back(cand) ;
        if(cand.candidate) m_repeat_candidates.push_back(pdig);
    }

    // erase repeats that are enclosed within other repeats 
    // ie that have an ancestor which is also a repeat candidate

    m_repeat_candidates.erase(
         std::remove_if(m_repeat_candidates.begin(), m_repeat_candidates.end(), *this ),
         m_repeat_candidates.end()
    );

    std::cout << " (**) candidates fulfil repeat/vert cuts   "  << std::endl ;
    std::cout << " (##) selected survive contained-repeat disqualification " << std::endl ;

    unsigned num_cands = cands.size() ;
    unsigned dmax = 20u ; 

    for(unsigned i=0 ; i < std::min(num_cands, dmax) ; i++)
    {
        NRepeat& cand = cands[i];
        cand.select = cand.isListed(m_repeat_candidates) ;
        std::cout << cand.desc() << " " << meshmeta(cand.first->mesh) << std::endl;
    }

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

    if(m_verbosity > 1)
    LOG(info) << "NScene::dump_repeat_candidates"
              << " verbosity " << m_verbosity 
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


    if(m_verbosity > 1)
    std::cout
              << " idx "    << std::setw(3) << idx 
              << " pdig "   << std::setw(32) << pdig
              << " nprog "  << std::setw(5) << num_progeny
              << " ninst " << std::setw(5) << num_instances
              << " mmeta " << meshmeta(mesh_id)
              << std::endl 
              ; 

    bool prolific  = num_progeny > 0 ; 
    if(prolific && m_verbosity > 1 )
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



void NScene::dumpNd(unsigned nidx, const char* msg)
{

    nd* n = getNd(nidx);
    LOG(info) << msg 
              << " nidx " << nidx
              << ( n ? " node exists " : " NO SUCH NODE " )
              << " verbosity " << m_verbosity  ; 

    if(!n) return ; 

    unsigned mesh_id = n->mesh ; 

    std::cout << std::endl 
              << n->detail()
              << std::endl 
              << " mesh_id " << mesh_id 
              << " meshmeta " << meshmeta(mesh_id) 
              << std::endl 
              ;   


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

    //if(m_verbosity > 1)
    LOG(info)<<"NScene::labelTree" 
             << " label_count (non-zero ridx labelTree_r) " << m_label_count 
             << " num_repeat_candidates " << m_repeat_candidates.size()
             ;
}


void NScene::labelTree_r(nd* n, unsigned ridx)
{
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



