#pragma once
/**
SScene.h
=========

Canonical SScene instance is member of SSim 
and is instanciated by the SSim ctor. 
This SScene instance is sibling of the canonical 
stree instance. 

The SScene is populated via a surprisingly high level
call stack::

    G4CXOpticks::setGeometry 
    SSim::initSceneFromTree
    SScene::initFromTree  


To some extent this acts as a minimal sub-selection of the 
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
#include "SBitSet.h"

struct SScene
{
    static constexpr const char* RELDIR = "scene" ;
    static constexpr const char* MESHGROUP = "meshgroup" ;
    static constexpr const char* MESHMERGE = "meshmerge" ;
    static constexpr const char* FRAME = "frame" ;
    static constexpr const char* INST_TRAN = "inst_tran.npy" ;
    static constexpr const char* INST_COL3 = "inst_col3.npy" ;
    static constexpr const char* INST_INFO = "inst_info.npy" ;

    bool dump ;

    std::vector<const SMeshGroup*>    meshgroup ;
    std::vector<const SMesh*>         meshmerge ;
    std::vector<sfr>                  frame ; 

    std::vector<int4>                 inst_info ;  // compound solid level 

    std::vector<glm::tmat4x4<float>>  inst_tran ;  // instance level 
    std::vector<glm::tvec4<int32_t>>  inst_col3 ;

    static SScene* Load(const char* dir);
    SScene();
    void check() const ; 

    void initFromTree(const stree* st);

    void initFromTree_Remainder(  const stree* st);
    void initFromTree_Triangulate(const stree* st);
    void initFromTree_Global(const stree* st, char ridx_type, int ridx );

    void initFromTree_Factor(const stree* st);
    void initFromTree_Factor_(int ridx, const stree* st);
    void initFromTree_Node(SMeshGroup* mg, int ridx, const snode& node, const stree* st);
    void initFromTree_Instance (const stree* st);

    const SMeshGroup* getMeshGroup(int idx) const ; 
    const SMesh*      getMeshMerge(int idx) const ; 


    const SMesh* get_mm(int mmidx) const ;
    const float* get_mn(int mmidx) const ; 
    const float* get_mx(int mmidx) const ; 
    const float* get_ce(int mmidx) const ; 

    bool is_empty() const ; 
    std::string desc() const ;
    std::string descDetail() const ;
    std::string descSize() const ;
    std::string descInstInfo() const ;
    std::string descFrame() const ;
    std::string descRange() const ;

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
    void addFrame( const sfr& _f); 

    sfr getFrame(int _idx=-1) const ; 

    static SScene* CopySelect( const SScene* src, const SBitSet* elv ); 
};




inline SScene* SScene::Load(const char* dir)
{
    SScene* s = new SScene ;
    s->load(dir);
    s->check(); 
    return s ;
}

inline SScene::SScene()
    :
    dump(ssys::getenvbool("SScene_dump"))
{
}

/**
SScene::check
---------------

Checks:

1. more than 0 frames, these should be added by SScene::addFrames
2. consistency between meshmerge and meshgroup 

**/


inline void SScene::check() const 
{
    int num_frame = frame.size(); 
    bool num_frame_expect =  num_frame > 0  ;

    if(!num_frame_expect) std::cout 
         << "SScene::check"
         << " num_frame " << num_frame 
         << " num_frame_expect " << ( num_frame_expect ? "YES" : "NO " ) 
         << "\n" 
         ; 

    assert( num_frame_expect ); 
    assert( meshmerge.size() == meshgroup.size() ); 
}


/**
SScene::initFromTree
---------------------

Note the surprisingly high level call stack,
immediately after stree population by U4Tree::Create::

    G4CXOpticks::setGeometry
    SSim::initSceneFromTree
    SScene::initFromTree

Creating::

    (SMeshGroup)meshgroup
    (SMesh)meshmerge 


TODO: adding frames at SScene::initFromTree 
seems not the right place for it : would be better to do this from the
main when rendering for simpler changing of framespec. Avoiding the need
to rerun the integration in order to change framespec. 

HMM: could do both adding dynamic framespec from some other path 
(like framespec.txt in invoking directory) that get to with SHIFT+NUMBER_KEY ? 

**/


inline void SScene::initFromTree(const stree* st)
{
    initFromTree_Remainder(st);
    initFromTree_Factor(st);
    initFromTree_Triangulate(st);

    initFromTree_Instance(st); 

    addFrames("$SScene__initFromTree_addFrames", st ); 
}

inline void SScene::initFromTree_Remainder(const stree* st)
{
    [[maybe_unused]] int num_rem = st->get_num_remainder(); 
    assert( num_rem == 1 ); 
    int ridx = 0 ; 
    initFromTree_Global( st, 'R', ridx ); 
}
inline void SScene::initFromTree_Triangulate(const stree* st)
{
    int num_rem = st->get_num_remainder(); 
    int num_fac = st->get_num_factor(); 
    int num_tri = st->get_num_triangulated(); 

    assert( num_rem == 1 ); 
    assert( num_tri == 1 || num_tri == 0  ); 

    if(num_tri == 1 )
    {
        int ridx = num_rem + num_fac + 0 ; 
        initFromTree_Global( st, 'T', ridx ); 
    }
}


/**
SScene::initFromTree_Global
---------------------------


**/

inline void SScene::initFromTree_Global(const stree* st, char ridx_type, int ridx )
{
    assert( ridx_type == 'R' || ridx_type == 'T' ); 
    const std::vector<snode>* _nodes = st->get_node_vector(ridx_type)  ; 
    assert( _nodes ); 

    int num_node = _nodes->size() ;
    if(dump) std::cout
        << "[ SScene::initFromTree_Global"
        << " num_node " << num_node
        << " ridx_type " << ridx_type
        << " ridx " << ridx 
        << std::endl
        ;

    SMeshGroup* mg = new SMeshGroup ; 
    for(int i=0 ; i < num_node ; i++)
    {
        const snode& node = (*_nodes)[i];
        initFromTree_Node(mg, ridx, node, st);
    }
    const SMesh* _mesh = SMesh::Concatenate( mg->subs, ridx );
    meshmerge.push_back(_mesh);
    meshgroup.push_back(mg);

    if(dump) std::cout
        << "] SScene::initFromTree_Global"
        << " num_node " << num_node 
        << " ridx_type " << ridx_type
        << " ridx " << ridx 
        << std::endl
        ;
}

inline void SScene::initFromTree_Factor(const stree* st)
{
    int num_rem = st->get_num_remainder(); 
    assert( num_rem == 1 ); 

    int num_fac = st->get_num_factor(); 
    for(int i=0 ; i < num_fac ; i++) initFromTree_Factor_(num_rem+i, st); 
}

/**
SScene::initFromTree_Factor_
-----------------------------

Note that the meshmerge and meshgroup contain the same triangulated
geometry info organized differently 

meshgroup
    SMeshGroup instances that maintain separate SMesh for each "Prim"
    (used by triangulated OptiX?)

meshmerge
    concatenated SMesh used by OpenGL 
    HMM: the concatenation could be deferred or redone following 
    lvid based sub-selection applied to mg->subs

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

1. get transform for snode::index from stree
2. import SMesh for snode::lvid from stree, possibly applying the transform
3. collect SMesh into SMeshGroup


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
    //strid::NarrowClear( inst_tran, st->inst ); // copy and narrow from st->inst into inst_tran 

    strid::NarrowDecodeClear(inst_tran, inst_col3, st->inst ); 

}



inline const SMeshGroup* SScene::getMeshGroup(int idx) const 
{
    return idx < int(meshgroup.size()) ? meshgroup[idx] : nullptr ; 
}
inline const SMesh*      SScene::getMeshMerge(int idx) const 
{
    return idx < int(meshmerge.size()) ? meshmerge[idx] : nullptr ; 
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


inline bool SScene::is_empty() const 
{
    return meshmerge.size() == 0 && meshgroup.size() == 0 && inst_info.size() == 0 && inst_tran.size() == 0 ; 
} 


inline std::string SScene::desc() const 
{
    std::stringstream ss ; 
    ss << "[ SScene::desc \n" ;
    ss << " is_empty " << ( is_empty() ? "YES" : "NO " ) << "\n" ; 
    ss << descSize() ; 
    ss << descInstInfo() ; 
    ss << descFrame() ; 
    ss << "] SScene::desc \n" ; 
    std::string str = ss.str(); 
    return str ; 
}

inline std::string SScene::descDetail() const 
{
    std::stringstream ss ; 
    ss << "[ SScene::detail \n" ;
    ss << descRange() ; 
    ss << "] SScene::detail \n" ; 
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
       << " inst_col3 " << inst_col3.size()
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
    [[maybe_unused]] int num_mg = meshgroup.size(); 
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
    NP* _inst_col3 = NPX::ArrayFromVec<int,glm::tvec4<int32_t>>( inst_col3, 4 ) ; 


    NPFold* fold = new NPFold ; 
    fold->add_subfold( MESHMERGE, _meshmerge ); 
    fold->add_subfold( MESHGROUP, _meshgroup ); 
    fold->add_subfold( FRAME,     _frame ); 
    fold->add( INST_INFO, _inst_info );
    fold->add( INST_TRAN, _inst_tran );
    fold->add( INST_COL3, _inst_col3 );

    return fold ; 
}
inline void SScene::import(const NPFold* fold)
{
    if(fold == nullptr) std::cerr << "SScene::import called with NULL fold argument\n" ; 
    if(fold == nullptr) return ; 

    const NPFold* _meshmerge = fold->get_subfold(MESHMERGE ); 
    const NPFold* _meshgroup = fold->get_subfold(MESHGROUP ); 
    const NPFold* _frame     = fold->get_subfold(FRAME ); 
    import_meshmerge( _meshmerge ); 
    import_meshgroup( _meshgroup ); 
    import_frame(     _frame ); 

    const NP* _inst_info = fold->get(INST_INFO); 
    const NP* _inst_tran = fold->get(INST_TRAN); 
    const NP* _inst_col3 = fold->get(INST_COL3); 

    stree::ImportArray<glm::tmat4x4<float>, float>( inst_tran, _inst_tran ); 
    stree::ImportArray<int4, int>( inst_info, _inst_info ); 
    stree::ImportArray<glm::tvec4<int32_t>, int>( inst_col3, _inst_col3 ); 
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
4. add last frame f0 fabricated from the ce of the first global mergedmesh 

For integrated running a sensible place to configure this 
is next to the setting of the output geometry directory::

    export G4CXOpticks__SaveGeometry_DIR=$HOME/.opticks/GEOM/$GEOM
    export SScene__initFromTree_addFrames=$HOME/.opticks/GEOM/${GEOM}_framespec.txt


**/

inline void SScene::addFrames(const char* path, const stree* st)
{
    std::string framespec ; 
    bool framespec_exists = spath::Read( framespec, path ); 

    if(framespec_exists)
    {
        std::vector<std::string> lines ; 
        sstr::SplitTrimSuppress(framespec.c_str(), '\n', lines) ; 

        int num_line = lines.size();
        for(int i=0 ; i < num_line ; i++)
        {
            const std::string& line = lines[i]; 
            std::string _spec = sstr::StripComment(line);  
            const char* spec = _spec.c_str(); 
            if(spec == nullptr) continue ; 
            if(strlen(spec) == 0) continue ;  

            bool has_frame = st->has_frame(spec);
            if(!has_frame)
            {
                std::cout 
                    << "SScene::addFrames FAIL to find frame " 
                    << " spec [" << ( spec ? spec : "-" ) << "]\n"
                    << " line [" << line << "]\n"  
                    ; 
                continue ;
            } 
 
            sfr f = st->get_frame(spec); 
            addFrame(f);  
        }
    }
    else
    {
        std::cout << "SScene::addFrames framespec path [" << ( path ? path : "-" ) << "] does not exist \n" ; 
    }

    // last frame that ensures always at least one  
    const float* _ce = get_ce(0) ; 
    sfr f0 = sfr::MakeFromCE(_ce) ;
    f0.set_name("MakeFromCE0");  
    addFrame(f0);  



    /**
    // this is wrong place to do this, need to do from main 
    // because want to easily change target

    const char* MOI = ssys::getenvvar("MOI", nullptr); 
    if(MOI)
    {
        sfr fm = st->get_frame(MOI); 
        addFrame(fm); 
    } 
    **/


}


inline void SScene::addFrame( const sfr& _f)
{
   sfr f = _f ; 
   f.set_idx( frame.size() ); 
   frame.push_back(f);  
}


/**
SScene::getFrame
----------------

Returns the *_idx* frame

For argument _idx beyond available frames returns the last frame

**/

inline sfr SScene::getFrame(int _idx) const
{
    int num_frame = frame.size(); 
    bool num_frame_expect =  num_frame > 0  ;
    int idx = ( _idx > -1 && _idx < num_frame ) ? _idx : num_frame-1  ; 

    if(!num_frame_expect) std::cout 
         << "SScene::getFrame"
         << " num_frame " << num_frame 
         << " num_frame_expect " << ( num_frame_expect ? "YES" : "NO " ) 
         << " _idx " << _idx
         << " idx " << idx
         << "\n" 
         ; 

    assert( num_frame_expect ); 

    const sfr& f = frame[idx] ; 

 
    assert( f.get_idx() == idx ); 
    return f ; 
}


inline SScene* SScene::CopySelect( const SScene* src, const SBitSet* elv ) // static
{
    SScene* dst = new SScene ; 
    size_t s_num_mg = src->meshgroup.size() ;  
    for(size_t i=0 ; i < s_num_mg ; i++)
    {
        const SMeshGroup* s_mg = src->meshgroup[i] ; 
        SMeshGroup* d_mg = s_mg->copy(elv) ;  
        if( d_mg == nullptr ) continue ;   // null when no subs are ELV selected

        dst->meshgroup.push_back(d_mg); 

        int ridx = i ;  // first gen assumption 
        const SMesh* d_mesh = SMesh::Concatenate( d_mg->subs, ridx );
        dst->meshmerge.push_back(d_mesh);
    }

    dst->frame = src->frame ;  

    /*
    size_t s_num_inst_info = src->inst_info.size() ; 
    size_t s_num_inst_tran = src->inst_tran.size() ; 
    */
    // TODO: use newly added inst_col3 info to apply selection to the inst_tran 
    //       hmm tri nodes are not instanced ? 

    return dst ; 
}



