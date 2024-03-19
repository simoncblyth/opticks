#pragma once
/**
SScene.h
=========

::

    ~/o/sysrap/tests/SScene_test.sh 


* Start triangle only 
* Needs to do equivalent of the non-OptiX parts of sutil::Scene
  (dont want to duplicate stree.h, so can just hold device pointers
   of uploaded arrays) 

* HOW MUCH AT SMesh.h LEVEL VS SScene.h LEVEL ? 

* OpenGL/CUDA interop-ing the triangle data is possible (but not straight off)


How to organize the instance transforms ?
-------------------------------------------

* tree inst are in contiguous blocks for each gas/mesh referenced 
* so can have single OpenGL SScene:inst upload referenced by (offset,count) for each merged mesh 


**/

#include "stree.h"
#include "SMesh.h"

struct SScene
{
    static constexpr const char* MESH_GRUP = "mesh_grup" ;
    static constexpr const char* INST_TRAN = "inst_tran.npy" ;
    static constexpr const char* INST_INFO = "inst_info.npy" ;

    bool dump ;

    std::vector<const SMesh*>         mesh_grup ;
    std::vector<int4>                 inst_info ; 
    std::vector<glm::tmat4x4<float>>  inst_tran ;

    static SScene* Load(const char* dir);
    SScene();

    void initFromTree(const stree* st);
    void initFromTree_Remainder(const stree* st);
    void initFromTree_Factor(const stree* st);
    void initFromTree_Factor_(int ridx, const stree* st);
    void initFromTree_Node(std::vector<const SMesh*>& subs, int ridx, const snode& node, const stree* st);
    void initFromTree_Instance (const stree* st);

    std::string descSize() const ;
    std::string descInstInfo() const ;
    std::string desc() const ;

    NPFold* serialize_mesh_grup() const ;
    void import_mesh_grup(const NPFold* _mesh_grup ) ; 
    NPFold* serialize() const ;
    void import(const NPFold* fold);

    void save(const char* dir) const ;
    void load(const char* dir);

};



inline SScene* SScene::Load(const char* dir)
{
    SScene* s = new SScene ;
    s->load(dir);
    return s ;
}

inline SScene::SScene()
    :
    dump(false)
{
}

inline void SScene::initFromTree(const stree* st)
{
    initFromTree_Remainder(st);
    initFromTree_Factor(st);
    initFromTree_Instance(st); 
}

inline void SScene::initFromTree_Remainder(const stree* st)
{
    int num_node = st->rem.size() ;
    if(dump) std::cout
        << "[ SScene::initFromTree_Remainder"
        << " num_node " << num_node
        << std::endl
        ;

    std::vector<const SMesh*> subs ;
    for(int i=0 ; i < num_node ; i++)
    {
        const snode& node = st->rem[i];
        initFromTree_Node(subs, 0, node, st);
    }
    const SMesh* _mesh = SMesh::Concatenate( subs, 0 );
    mesh_grup.push_back(_mesh);

    if(dump) std::cout
        << "] SScene::initFromTree_Remainder"
        << " num_node " << num_node 
        << std::endl
        ;
}

inline void SScene::initFromTree_Factor(const stree* st)
{
    int num_fac = st->get_num_factor(); 
    for(int i=0 ; i < num_fac ; i++) initFromTree_Factor_(1+i, st); 
}

inline void SScene::initFromTree_Factor_(int ridx, const stree* st)
{
    assert( ridx > 0 ); 
    int q_repeat_index = ridx ; 
    int q_repeat_ordinal = 0 ;   // just first repeat 
    std::vector<snode> nodes ; 
    st->get_repeat_node(nodes, q_repeat_index, q_repeat_ordinal) ; 
    int num_node = nodes.size(); 

    if(dump) std::cout 
       << "SScene::initFromTree_Factor"
       << " ridx " << ridx
       << " num_node " << num_node
       << std::endl 
       ;

    std::vector<const SMesh*> subs ; 
    for(int i=0 ; i < num_node ; i++)
    {
        const snode& node = nodes[i]; 
        initFromTree_Node(subs, ridx, node, st); 
    }
    const SMesh* _mesh = SMesh::Concatenate( subs, ridx ); 
    mesh_grup.push_back(_mesh); 
}

/**
SScene::initFromTree_Node
---------------------------

Note that some snode within sfactor sub trees have 
different transforms relative to the outer snode
such as PMT innards. Hence the SMesh vtx 
are flattened into the instance frame.

Recall that instance outer nodes will need no transform, 
but for volumes within the subtree some relative 
transforms will in general need to be applied such as PMT innards
 
Constrast with CSGFoundry 
which goes further delving into the transforms of the 
CSG constituents.

**/

inline void SScene::initFromTree_Node(std::vector<const SMesh*>& submesh, int ridx, const snode& node, const stree* st)
{
    glm::tmat4x4<double> m2w(1.);  
    glm::tmat4x4<double> w2m(1.);  
    bool local = true ;        // WHAT ABOUT FOR REMAINDER NODES ?
    bool reverse = false ;     // ?
    st->get_node_product(m2w, w2m, node.index, local, reverse, nullptr ); 
    bool is_identity_m2w = stra<double>::IsIdentity(m2w) ; 

    const char* so = st->soname[node.lvid].c_str();  // raw (not 0x stripped) name
    const NPFold* fold = st->mesh->get_subfold(so)  ;
    assert(fold); 

    const SMesh* _mesh = SMesh::Import( fold, &m2w ); 
    submesh.push_back(_mesh); 

    if(dump) std::cout 
       << "SScene::initFromTree_Node"
       << " node.lvid " << node.lvid 
       << " st.soname[node.lvid] " << st->soname[node.lvid] 
       << " _mesh " <<  _mesh->brief()  
       << " is_identity_m2w " << ( is_identity_m2w ? "YES" : "NO " )
       << std::endl 
       ;

    if(dump && !is_identity_m2w) std::cout << _mesh->descTransform() << std::endl ;  
}

inline void SScene::initFromTree_Instance(const stree* st)
{
    inst_info = st->inst_info ; 
    strid::NarrowClear( inst_tran, st->inst ); 
}


inline std::string SScene::desc() const 
{
    std::stringstream ss ; 
    ss << descSize() ; 
    ss << descInstInfo() ; 
    std::string str = ss.str(); 
    return str ; 
}

inline std::string SScene::descSize() const 
{
    std::stringstream ss ; 
    ss << "SScene::descSize"
       << " mesh_grup " << mesh_grup.size()
       << " inst_info " << inst_info.size()
       << " inst_tran " << inst_tran.size()
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}

inline std::string SScene::descInstInfo() const
{
    std::stringstream ss ;
    ss << "[SScene::descInstInfo {ridx, inst_count, inst_offset, 0} " << std::endl ; 
    int num_inst_info = inst_info.size(); 
    for(int i=0 ; i < num_inst_info ; i++)
    {
        const int4& info = inst_info[i] ; 
        ss 
           << "{" 
           << std::setw(3) << info.x
           << "," 
           << std::setw(7) << info.y
           << "," 
           << std::setw(7) << info.z 
           << "," 
           << std::setw(3) << info.w 
           << "}"
           << std::endl  
           ;
    }
    ss << "]SScene::descInstInfo" << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}


inline NPFold* SScene::serialize_mesh_grup() const 
{
    NPFold* _mesh_grup = new NPFold ; 
    int num_mesh_grup = mesh_grup.size(); 
    for(int i=0 ; i < num_mesh_grup ; i++)
    {
        const SMesh* m = mesh_grup[i] ; 
        _mesh_grup->add_subfold( m->name, m->serialize() ); 
    } 
    return _mesh_grup ; 
}

inline void SScene::import_mesh_grup(const NPFold* _mesh_grup ) 
{
    int num_mesh_grup = _mesh_grup->get_num_subfold();
    std::cout << "SScene::import_mesh_grup num_mesh_grup " << num_mesh_grup << std::endl ;     
    for(int i=0 ; i < num_mesh_grup ; i++)
    {
        const NPFold* sub = _mesh_grup->get_subfold(i); 
        const SMesh* m = SMesh::Import(sub) ;  
        mesh_grup.push_back(m); 
    }
}

inline NPFold* SScene::serialize() const 
{
    NPFold* _mesh_grup = serialize_mesh_grup() ;
    NP* _inst_tran = NPX::ArrayFromVec<float, glm::tmat4x4<float>>( inst_tran, 4, 4) ;
    NP* _inst_info = NPX::ArrayFromVec<int,int4>( inst_info, 4 ) ; 

    NPFold* fold = new NPFold ; 
    fold->add_subfold( MESH_GRUP, _mesh_grup ); 
    fold->add( INST_INFO, _inst_info );
    fold->add( INST_TRAN, _inst_tran );

    return fold ; 
}
inline void SScene::import(const NPFold* fold)
{
    const NPFold* _mesh_grup = fold->get_subfold(MESH_GRUP ); 
    import_mesh_grup( _mesh_grup ); 

    const NP* _inst_info = fold->get(INST_INFO); 
    const NP* _inst_tran = fold->get(INST_TRAN); 

    stree::ImportArray<glm::tmat4x4<float>, float>( inst_tran, _inst_tran ); 
    stree::ImportArray<int4, int>( inst_info, _inst_info ); 
}

inline void SScene::save(const char* dir) const 
{
    NPFold* fold = serialize(); 
    fold->save(dir); 
}
inline void SScene::load(const char* dir) 
{
    NPFold* fold = NPFold::Load(dir); 
    import(fold); 
}

