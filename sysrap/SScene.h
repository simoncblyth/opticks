#pragma once
/**
SScene.h
=========

To some extent this acts as a minimal selection of the 
full stree.h info needed to render

::

    ~/o/sysrap/tests/SScene_test.sh 

* OpenGL/CUDA interop-ing the triangle data is possible (but not straight off)

* TODO: solid selection eg skipping virtuals so can see PMT shapes 

* WIP: incorporate into standard workflow

  * treat SScene.h as sibling to stree.h within SSim.hh 
  * invoke the SScene.h creation from stree immediately after stree creation by U4Tree 

**/

#include "ssys.h"
#include "stree.h"
#include "SMesh.h"
#include "SMeshGroup.h"

struct SScene
{
    static constexpr const char* RELDIR = "scene" ;
    static constexpr const char* MESHGROUP = "meshgroup" ;
    static constexpr const char* MESHMERGE = "meshmerge" ;
    static constexpr const char* FRAME = "frame" ;
    static constexpr const char* INST_TRAN = "inst_tran.npy" ;
    static constexpr const char* INST_INFO = "inst_info.npy" ;

    bool dump ;

    std::vector<const SMeshGroup*>    meshgroup ;
    std::vector<const SMesh*>         meshmerge ;
    std::vector<sfr>                  frame ; 

    std::vector<int4>                 inst_info ; 
    std::vector<glm::tmat4x4<float>>  inst_tran ;

    static SScene* Load(const char* dir);
    SScene();
    void check() const ; 

    void initFromTree(const stree* st);
    void initFromTree_Remainder(const stree* st);
    void initFromTree_Factor(const stree* st);
    void initFromTree_Factor_(int ridx, const stree* st);
    void initFromTree_Node(SMeshGroup* mg, int ridx, const snode& node, const stree* st);
    void initFromTree_Instance (const stree* st);

    const SMesh* get_mm(int mmidx) const ;
    const float* get_mn(int mmidx) const ; 
    const float* get_mx(int mmidx) const ; 
    const float* get_ce(int mmidx) const ; 

    std::string descSize() const ;
    std::string descInstInfo() const ;
    std::string descFrame() const ;
    std::string descRange() const ;
    std::string desc() const ;

    NPFold* serialize_meshmerge() const ;
    void import_meshmerge(const NPFold* _meshmerge ) ; 

    NPFold* serialize_meshgroup() const ;
    void import_meshgroup(const NPFold* _meshgroup ) ; 

    NPFold* serialize_frame() const ;
    void import_frame(const NPFold* _frame ) ; 

    NPFold* serialize() const ;
    void import(const NPFold* fold);

    void save(const char* dir) const ;
    void load(const char* dir);

    void addFrames(const char* path, const stree* st); 
    sfr getFrame(int _idx=-1) const ; 

};




inline SScene* SScene::Load(const char* dir)
{
    SScene* s = new SScene ;
    s->load(dir);
    return s ;
}

inline SScene::SScene()
    :
    dump(ssys::getenvbool("SScene_dump"))
{
}

inline void SScene::check() const 
{
    assert( meshmerge.size() == meshgroup.size() ); 
}

inline void SScene::initFromTree(const stree* st)
{
    initFromTree_Remainder(st);
    initFromTree_Factor(st);
    initFromTree_Instance(st); 

    addFrames("$SScene__initFromTree_addFrames", st ); 
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

/**
SScene::initFromTree_Factor_
-----------------------------

Note that the meshmerge and meshgroup contain the same triangulated
geometry info organized differently 

meshgroup
    SMeshGroup instances that maintain separate SMesh for each "Prim"

meshmesh
    concatenated SMesh used by OpenGL 

**/

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

Note that some snode within sfactor subtrees have 
different transforms relative to the outer snode
such as PMT innards. Hence the SMesh vtx 
are flattened into the instance frame.

Recall that instance outer nodes will need no transform, 
but for volumes within the subtree some relative 
transforms will in general need to be applied.
 
Contrast with CSGFoundry which goes further delving into 
the transforms of the CSG constituents.

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
    // st->soname is now stripped at collection by stree.h with sstr::StripTail_Unique 

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


inline const SMesh* SScene::get_mm(int mmidx) const 
{
    const SMesh* mm = mmidx < int(meshmerge.size()) ? meshmerge[mmidx] : nullptr ; 
    return mm ; 
}
inline const float* SScene::get_mn(int mmidx) const 
{
    const SMesh* mm = get_mm(mmidx); 
    return mm ? mm->get_mn() : nullptr ;  
} 
inline const float* SScene::get_mx(int mmidx) const 
{
    const SMesh* mm = get_mm(mmidx); 
    return mm ? mm->get_mx() : nullptr ;  
} 
inline const float* SScene::get_ce(int mmidx) const 
{
    const SMesh* mm = get_mm(mmidx); 
    return mm ? mm->get_ce() : nullptr ;  
} 




inline std::string SScene::desc() const 
{
    std::stringstream ss ; 
    ss << "[ SScene::desc \n" ; 
    ss << descSize() ; 
    ss << descInstInfo() ; 
    ss << descFrame() ; 
    ss << descRange() ; 
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

inline std::string SScene::descFrame() const
{
    int num_frame = frame.size(); 
    std::stringstream ss ;
    ss << "[SScene::descFrame num_frame " << num_frame << std::endl ; 
    for(int i=0 ; i < num_frame ; i++) 
    {
        const sfr& f = frame[i]; 
        ss << f.desc() ; 
    }
    ss << "]SScene::descFrame num_frame " << num_frame << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}

inline std::string SScene::descRange() const
{
    int num_mm = meshmerge.size(); 
    int num_mg = meshgroup.size(); 
    assert( num_mm == num_mg ); 
    int num = num_mm ; 

    std::stringstream ss ;
    ss << "[SScene::descRange num " << num << std::endl ; 
    for(int i=0 ; i < num ; i++) 
    {
        const SMeshGroup* mg = meshgroup[i] ; 
        const SMesh* mm = meshmerge[i] ;
        ss << "mg[" << i << "]\n" << mg->descRange() << "\n" ; 
        ss << "mm[" << i << "]\n" << mm->descRange() << "\n" ; 
    }
    ss << "]SScene::descRange num " << num << std::endl ; 
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


inline NPFold* SScene::serialize_frame() const 
{
    NPFold* _frame = new NPFold ; 
    int num_frame = frame.size(); 
    for(int i=0 ; i < num_frame ; i++)
    {
        const sfr& f  = frame[i] ; 
        std::string key = f.get_key() ; 
        _frame->add( key.c_str(), f.serialize() ); 
    } 
    return _frame ; 
}

inline void SScene::import_frame(const NPFold* _frame ) 
{
    int num_frame = _frame ? _frame->num_items() : 0 ;
    for(int i=0 ; i < num_frame ; i++)
    {
        const NP* a = _frame->get_array(i); 
        sfr f = sfr::Import(a) ;  
        frame.push_back(f); 
    }
}



inline NPFold* SScene::serialize() const 
{
    NPFold* _meshmerge = serialize_meshmerge() ;
    NPFold* _meshgroup = serialize_meshgroup() ;
    NPFold* _frame     = serialize_frame() ;
    NP* _inst_tran = NPX::ArrayFromVec<float, glm::tmat4x4<float>>( inst_tran, 4, 4) ;
    NP* _inst_info = NPX::ArrayFromVec<int,int4>( inst_info, 4 ) ; 

    NPFold* fold = new NPFold ; 
    fold->add_subfold( MESHMERGE, _meshmerge ); 
    fold->add_subfold( MESHGROUP, _meshgroup ); 
    fold->add_subfold( FRAME,     _frame ); 
    fold->add( INST_INFO, _inst_info );
    fold->add( INST_TRAN, _inst_tran );

    return fold ; 
}
inline void SScene::import(const NPFold* fold)
{
    const NPFold* _meshmerge = fold->get_subfold(MESHMERGE ); 
    const NPFold* _meshgroup = fold->get_subfold(MESHGROUP ); 
    const NPFold* _frame     = fold->get_subfold(FRAME ); 
    import_meshmerge( _meshmerge ); 
    import_meshgroup( _meshgroup ); 
    import_frame(     _frame ); 

    const NP* _inst_info = fold->get(INST_INFO); 
    const NP* _inst_tran = fold->get(INST_TRAN); 

    stree::ImportArray<glm::tmat4x4<float>, float>( inst_tran, _inst_tran ); 
    stree::ImportArray<int4, int>( inst_info, _inst_info ); 
}

inline void SScene::save(const char* dir) const 
{
    NPFold* fold = serialize(); 
    fold->save(dir, RELDIR); 
}
inline void SScene::load(const char* dir) 
{
    NPFold* fold = NPFold::Load(dir, RELDIR); 
    import(fold); 
}


/**
SScene::addFrames
------------------

Canonically called from SScene::initFromTree with path 
argument from envvar::

    SScene__initFromTree_addFrames

Which is set for example from::

    ~/o/sysrap/tests/SScene_test.sh


1. read framespec string from path file
2. parse the string splitting into trimmed lines
3. for each line get sfr with stree::get_frame add to frame vector

**/

inline void SScene::addFrames(const char* path, const stree* st)
{
    std::string framespec ; 
    bool read = spath::Read( framespec, path ); 
    if(!read) return ; 

    std::vector<std::string> lines ; 
    sstr::SplitTrimSuppress(framespec.c_str(), '\n', lines) ; 

    int num_line = lines.size(); 
    for(int i=0 ; i < num_line ; i++)
    {
        const std::string& line = lines[i]; 
        sfr f = st->get_frame(line.c_str());  
        frame.push_back(f); 
    }
}


inline sfr SScene::getFrame(int _idx) const
{
    int num_frame = frame.size(); 
    int idx = ( _idx > -1 && _idx < num_frame ) ? _idx : -1 ; 

    const float* _ce = get_ce(0) ; 
    std::cout 
         << "SScene::getFrame"
         << " num_frame " << num_frame 
         << " _idx " << _idx
         << " idx " << idx
         << " _ce[3] " << ( _ce ? _ce[3] : -1.f )    
         << "\n" 
         ; 
  
    sfr fr = idx == -1 ? sfr::MakeFromCE(_ce) : frame[idx] ; 
    fr.set_idx(idx); 
    return fr ; 
}

