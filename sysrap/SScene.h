#pragma once
/**
SScene.h
=========

::

    ~/o/sysrap/tests/SScene_test.sh 

* OpenGL/CUDA interop-ing the triangle data is possible (but not straight off)

**/

#include "stree.h"
#include "SMesh.h"
#include "SMeshGroup.h"

struct SScene
{
    static constexpr const char* MESHGROUP = "meshgroup" ;
    static constexpr const char* MESHMERGE = "meshmerge" ;
    static constexpr const char* INST_TRAN = "inst_tran.npy" ;
    static constexpr const char* INST_INFO = "inst_info.npy" ;

    bool dump ;

    std::vector<const SMeshGroup*>    meshgroup ;
    std::vector<const SMesh*>         meshmerge ; // formerly mesh_grup

    std::vector<int4>                 inst_info ; 
    std::vector<glm::tmat4x4<float>>  inst_tran ;

    static SScene* Load(const char* dir);
    SScene();

    void initFromTree(const stree* st);
    void initFromTree_Remainder(const stree* st);
    void initFromTree_Factor(const stree* st);
    void initFromTree_Factor_(int ridx, const stree* st);
    void initFromTree_Node(SMeshGroup* mg, int ridx, const snode& node, const stree* st);
    void initFromTree_Instance (const stree* st);

    std::string descSize() const ;
    std::string descInstInfo() const ;
    std::string desc() const ;

    NPFold* serialize_meshmerge() const ;
    void import_meshmerge(const NPFold* _meshmerge ) ; 

    NPFold* serialize_meshgroup() const ;
    void import_meshgroup(const NPFold* _meshgroup ) ; 

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

    SMeshGroup* mg = new SMeshGroup ; 
    int ridx = 0 ; 
    for(int i=0 ; i < num_node ; i++)
    {
        const snode& node = st->rem[i];
        initFromTree_Node(mg, ridx, node, st);
    }
    const SMesh* _mesh = SMesh::Concatenate( mg->subs, 0 );
    meshmerge.push_back(_mesh);
    meshgroup.push_back(mg);

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

    SMeshGroup* mg = new SMeshGroup ; 
    for(int i=0 ; i < num_node ; i++)
    {
        const snode& node = nodes[i]; 
        initFromTree_Node(mg, ridx, node, st); 
    }
    const SMesh* _mesh = SMesh::Concatenate( mg->subs, ridx ); 
    meshmerge.push_back(_mesh); 
    meshgroup.push_back(mg); 
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

inline void SScene::initFromTree_Node(SMeshGroup* mg, int ridx, const snode& node, const stree* st)
{
    glm::tmat4x4<double> m2w(1.);  
    glm::tmat4x4<double> w2m(1.);  
    bool local = true ;        // WHAT ABOUT FOR REMAINDER NODES ?
    bool reverse = false ;     // ?
    st->get_node_product(m2w, w2m, node.index, local, reverse, nullptr ); 
    bool is_identity_m2w = stra<double>::IsIdentity(m2w) ; 

    const char* so = st->soname[node.lvid].c_str();  
    // raw (not 0x stripped) name : NO LONGER TRUE : UNIQUE STRIPPING DONE BY STREE

    const NPFold* fold = st->mesh->get_subfold(so)  ;
    assert(fold); 
    const SMesh* _mesh = SMesh::Import( fold, &m2w ); 

    mg->subs.push_back(_mesh); 
    mg->names.push_back(so);  


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
    ss << "[ SScene::desc \n" ; 
    ss << descSize() ; 
    ss << descInstInfo() ; 
    ss << "] SScene::desc \n" ; 
    std::string str = ss.str(); 
    return str ; 
}

inline std::string SScene::descSize() const 
{
    std::stringstream ss ; 
    ss << "SScene::descSize"
       << " meshmerge " << meshmerge.size()
       << " meshgroup " << meshgroup.size()
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
    int tot_inst = 0 ;  
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
        tot_inst += info.y ;
    }
    ss << "]SScene::descInstInfo tot_inst " << tot_inst << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}


inline NPFold* SScene::serialize_meshmerge() const 
{
    NPFold* _meshmerge = new NPFold ; 
    int num_meshmerge = meshmerge.size(); 
    for(int i=0 ; i < num_meshmerge ; i++)
    {
        const SMesh* m = meshmerge[i] ; 
        _meshmerge->add_subfold( m->name, m->serialize() ); 
    } 
    return _meshmerge ; 
}

inline NPFold* SScene::serialize_meshgroup() const 
{
    NPFold* _meshgroup = new NPFold ; 
    int num_meshgroup = meshgroup.size(); 
    for(int i=0 ; i < num_meshgroup ; i++)
    {
        const SMeshGroup* mg = meshgroup[i] ; 
        const char* name = SMesh::FormName(i); 
        _meshgroup->add_subfold( name, mg->serialize() ); 
    } 
    return _meshgroup ; 
}



inline void SScene::import_meshmerge(const NPFold* _meshmerge ) 
{
    int num_meshmerge = _meshmerge ? _meshmerge->get_num_subfold() : 0 ;
    for(int i=0 ; i < num_meshmerge ; i++)
    {
        const NPFold* sub = _meshmerge->get_subfold(i); 
        const SMesh* m = SMesh::Import(sub) ;  
        meshmerge.push_back(m); 
    }
}

inline void SScene::import_meshgroup(const NPFold* _meshgroup ) 
{
    int num_meshgroup = _meshgroup ? _meshgroup->get_num_subfold() : 0 ;
    for(int i=0 ; i < num_meshgroup ; i++)
    {
        const NPFold* sub = _meshgroup->get_subfold(i); 
        const SMeshGroup* mg = SMeshGroup::Import(sub) ;  
        meshgroup.push_back(mg); 
    }
}


inline NPFold* SScene::serialize() const 
{
    NPFold* _meshmerge = serialize_meshmerge() ;
    NPFold* _meshgroup = serialize_meshgroup() ;
    NP* _inst_tran = NPX::ArrayFromVec<float, glm::tmat4x4<float>>( inst_tran, 4, 4) ;
    NP* _inst_info = NPX::ArrayFromVec<int,int4>( inst_info, 4 ) ; 

    NPFold* fold = new NPFold ; 
    fold->add_subfold( MESHMERGE, _meshmerge ); 
    fold->add_subfold( MESHGROUP, _meshgroup ); 
    fold->add( INST_INFO, _inst_info );
    fold->add( INST_TRAN, _inst_tran );

    return fold ; 
}
inline void SScene::import(const NPFold* fold)
{
    const NPFold* _meshmerge = fold->get_subfold(MESHMERGE ); 
    const NPFold* _meshgroup = fold->get_subfold(MESHGROUP ); 
    import_meshmerge( _meshmerge ); 
    import_meshgroup( _meshgroup ); 

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

