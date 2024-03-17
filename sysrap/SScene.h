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

Inputs from stree.h 
---------------------

* st->mesh : NPFold of meshes of vtx and tri  "GEOM st" keyed on mesh (LV) names 

* HMM: the snode within sfactor might have different transforms ? 

  * need to flatten all snode vtx of the sfactor into the frame of the outer snode 
    (aka the instance transform) 

    * that means that vertices for the solid of the outer volume of an instance
      will need no transform, but for volumes within the subtree some relative 
      transforms will in general need to be applied
   
      * BUT : IN MOST CASES (PMTs) ALL THE RELATIVE TRANSFORMS WILL BE IDENTITY
   
        * NOT SO FOR PMT INNARDS : THEY ALL HAVE TRANSFORMS

  * HMM: CSGFoundry must do similar ? IT GOES FUTHER DELVING INTO CSG CONSTIUENTS
    WITH stree::get_combined_transfor

  * stree::get_node_product with local:true for the nidx of the snode



**/

#include "stree.h"
#include "SMesh.h"

struct SScene
{
    const stree* st ;
    std::vector<const SMesh*> mesh ; 

    SScene( const stree* _st ); 

    void init(); 
    void initRemainder();
    void initFactor();
    void initFactor_(int ridx);

    void initNode(std::vector<const SMesh*>& subs, int ridx, const snode& node); 

    std::string desc() const ; 

    NPFold* serialize() const ; 
    void save(const char* dir) const ; 

};

inline SScene::SScene( const stree* _st )
    :
    st(_st)
{
    init(); 
}

inline void SScene::init()
{
    initRemainder(); 
    initFactor(); 
}

inline void SScene::initRemainder()
{
    int num_node = st->rem.size() ; 
    std::cout
        << "[ SScene::initRemainder"
        << " num_node " << num_node 
        << std::endl
        ;
 
    std::vector<const SMesh*> subs ; 
    for(int i=0 ; i < num_node ; i++)
    {
        const snode& node = st->rem[i]; 
        initNode(subs, 0, node); 
    }
    const SMesh* _mesh = SMesh::Concatenate( subs, 0 ); 
    mesh.push_back(_mesh); 

    std::cout
        << "] SScene::initRemainder"
        << " num_node " << num_node 
        << std::endl
        ;
}

inline void SScene::initFactor()
{
    int num_fac = st->get_num_factor(); 
    for(int i=0 ; i < num_fac ; i++) initFactor_(1+i); 
}

inline void SScene::initFactor_(int ridx)
{
    assert( ridx > 0 ); 
    int q_repeat_index = ridx ; 
    int q_repeat_ordinal = 0 ;   // just first repeat 
    std::vector<snode> nodes ; 
    st->get_repeat_node(nodes, q_repeat_index, q_repeat_ordinal) ; 
    int num_node = nodes.size(); 

    std::cout 
       << "SScene::initFactor"
       << " ridx " << ridx
       << " num_node " << num_node
       << std::endl 
       ;

    std::vector<const SMesh*> subs ; 
    for(int i=0 ; i < num_node ; i++)
    {
        const snode& node = nodes[i]; 
        initNode(subs, ridx, node); 
    }
    const SMesh* _mesh = SMesh::Concatenate( subs, ridx ); 
    mesh.push_back(_mesh); 
}

/**
SScene::initNode
------------------

Need meshgroup, to keep separate lists of relative placed meshes ?
Or could keep flat and identify groups by ridx ? 

SMeshGroup with vectors of SMesh ? 

* suspect the sutil::Scene meshgroup corresponds more to different factors ?
* Add SMesh::upload SMeshGroup::upload with device pointers in SMesh 
* OpenGL favors effective mesh merging ... 
 
  * hmm maybe OptiX does too ?  
  * could implement SMesh concatenation 

    * vtx, nrm are trivial
    * tri needs index offsetting  (see old GMesh/GMergedMesh)

  * OpenGL/CUDA interop-ing the triangle data is possible (but not straight off)
  * can start with duplicated arrays : in anycase always need without OpenGL route 


**/

inline void SScene::initNode(std::vector<const SMesh*>& submesh, int ridx, const snode& node)
{
    glm::tmat4x4<double> m2w(1.);  
    glm::tmat4x4<double> w2m(1.);  
    bool local = true ;        // WHAT ABOUT FOR REMAINDER NODES ?
    bool reverse = false ;     // ?
    st->get_node_product(m2w, w2m, node.index, local, reverse, nullptr ); 
    bool is_identity_m2w = stra<double>::IsIdentity(m2w) ; 

    const char* so = st->soname[node.lvid].c_str();  // raw (not 0x stripped) name
    const NPFold* fold = st->mesh->get_subfold(so)  ;
    const SMesh* _mesh = SMesh::Import( fold, &m2w ); 
    submesh.push_back(_mesh); 

    std::cout 
       << " node.lvid " << node.lvid 
       << " st.soname[node.lvid] " << st->soname[node.lvid] 
       << " _mesh " <<  _mesh->brief()  
       << " is_identity_m2w " << ( is_identity_m2w ? "YES" : "NO " )
       << std::endl 
       ;

    if(!is_identity_m2w) std::cout << _mesh->descTransform() << std::endl ;  
}


inline std::string SScene::desc() const 
{
    std::stringstream ss ; 
    ss << "SScene::desc"
       ;
    std::string str = ss.str(); 
    return str ; 
}

inline NPFold* SScene::serialize() const 
{
    NPFold* fold = new NPFold ; 
    int num_mesh = mesh.size(); 
    for(int i=0 ; i < num_mesh ; i++)
    {
        const SMesh* m = mesh[i] ; 
        fold->add_subfold( m->name, m->serialize() ); 
    } 
    return fold ; 
}

inline void SScene::save(const char* dir) const 
{
    NPFold* fold = serialize(); 
    fold->save(dir); 
}




