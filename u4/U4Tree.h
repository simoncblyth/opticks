#pragma once
/**
U4Tree.h : explore minimal approach to geometry translation
==============================================================

TODO:

* naming 
* disqualify contained repeats
* maintain correspondence between source nodes and destination nodes thru the factorization
* transform combination   
* transform rebase

* split off stree.h snode.h 

**/


#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <sstream>

#include <glm/glm.hpp>

#include "sdigest.h"
#include "sfreq.h"

#include "G4VPhysicalVolume.hh"
#include "U4Transform.h"
#include "NP.hh"


class G4VPhysicalVolume ; 

struct snode
{
    static constexpr const int NV = 8 ; 

    int index ; 
    int depth ; 
    int sibdex ; 
    int parent ; 

    int num_child ; 
    int first_child ; 
    int next_sibling ; 
    int lvid ;

    std::string desc() const ; 
}; 


std::string snode::desc() const
{
    std::stringstream ss ; 
    ss << "snode"
       << " id:" << std::setw(7) << index
       << " de:" << std::setw(2) << depth 
       << " si:" << std::setw(5) << sibdex 
       << " pa:" << std::setw(7) << parent
       << " nc:" << std::setw(5) << num_child
       << " fc:" << std::setw(7) << first_child
       << " ns:" << std::setw(7) << next_sibling
       << " lv:" << std::setw(3) << lvid 
       ;
    std::string s = ss.str(); 
    return s ; 

}



struct stree
{
    static constexpr const char* NDS = "nds.npy" ; 
    static constexpr const char* TRS = "trs.npy" ; 
    static constexpr const char* SONAME = "soname.txt" ; 
    static constexpr const char* DIGS = "digs.txt" ; 
    static constexpr const char* SUBS = "subs.txt" ; 
    static constexpr const char* SUBS_FREQ = "subs_freq" ; 

    std::vector<std::string> soname ; 
    std::vector<glm::tmat4x4<double>> trs ; 
    std::vector<snode> nds ; 
    std::vector<std::string> digs ; // single node digest  
    std::vector<std::string> subs ; // subtree digest 
    sfreq* subs_freq ; 

    stree(); 

    std::string desc() const ; 
    static std::string Digest(int lvid, const glm::tmat4x4<double>& tr ); 
    static std::string Digest(int lvid );

    void get_children(std::vector<int>& children, int nidx) const ; 
    void get_progeny_r( std::vector<int>& progeny, int nidx ) const ; 

    int get_parent(int nidx) const ; 
    void get_ancestors( std::vector<int>& ancestors, int nidx ) const ; 
    void get_nodes(std::vector<int>& nodes, const char* sub) const ; 


    void classifySubtrees(); 
    std::string subtree_digest( int nidx ) const ;  
    std::string desc_node(int nidx) const ; 
    std::string desc_nodes(const std::vector<int>& vnidx, unsigned edgeitems=10) const ; 

    void save( const char* fold ) const ; 
    void load( const char* fold ); 
};

inline stree::stree()
    :
    subs_freq(new sfreq)
{
}

inline std::string stree::desc() const 
{
    std::stringstream ss ; 
    ss << "stree::desc"
       << " nds " << nds.size()
       << " trs " << trs.size()
       << " digs " << digs.size()
       << " subs " << subs.size()
       << " soname " << soname.size()
       << " subs_freq " << std::endl 
       << ( subs_freq ? subs_freq->desc() : "-" )
       << std::endl 
       ;

    std::string s = ss.str(); 
    return s ; 
}


/**
stree::Digest
----------------------

Progeny digest needs to ncompassing transforms + lvid of subnodes, but only lvid of 
the node in question ?  

**/

inline std::string stree::Digest(int lvid, const glm::tmat4x4<double>& tr ) // static
{
    sdigest u ; 
    u.add( lvid ); 
    u.add( (char*)glm::value_ptr(tr), sizeof(double)*16 ) ; 
    std::string dig = u.finalize(); 
    return dig ; 
} 

inline std::string stree::Digest(int lvid ) // static
{
    return sdigest::Int(lvid); 
}

inline void stree::get_children( std::vector<int>& children , int nidx ) const 
{
    const snode& nd = nds[nidx]; 
    assert( nd.index == nidx ); 

    int ch = nd.first_child ; 
    while( ch > -1 )
    {
        const snode& child = nds[ch] ; 
        assert( child.parent == nd.index ); 
        children.push_back(child.index); 
        ch = child.next_sibling ; 
    }
    assert( int(children.size()) == nd.num_child ); 
}

inline void stree::get_progeny_r( std::vector<int>& progeny , int nidx ) const 
{
    std::vector<int> children ; 
    get_children(children, nidx); 
    std::copy(children.begin(), children.end(), std::back_inserter(progeny)); 
    for(unsigned i=0 ; i < children.size() ; i++) get_progeny_r(progeny, children[i] );
}

inline int stree::get_parent(int nidx) const { return nidx > -1 ? nds[nidx].parent : -1 ; }

inline void stree::get_ancestors( std::vector<int>& ancestors, int nidx ) const 
{
    int parent = get_parent(nidx) ; 
    while( parent > -1 )
    {
        ancestors.push_back(parent); 
        parent = get_parent(parent); 
    } 
    std::reverse( ancestors.begin(), ancestors.end() ); 
}

inline void stree::get_nodes(std::vector<int>& nodes, const char* sub) const 
{
    for(unsigned i=0 ; i < subs.size() ; i++) if(strcmp(subs[i].c_str(), sub)==0) nodes.push_back(int(i)) ;  
}

inline void stree::classifySubtrees()
{
    std::cout << "[ stree::classifySubtrees " << std::endl ; 
    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++) 
    {
        std::string sub = subtree_digest(nidx) ;
        subs.push_back(sub) ; 
        subs_freq->add(sub.c_str());  
    }
    subs_freq->sort(); 
    std::cout << "] stree::classifySubtrees " << std::endl ; 
}

inline std::string stree::subtree_digest(int nidx) const 
{
    std::vector<int> progeny ; 
    get_progeny_r(progeny, nidx); 

    sdigest u ;  
    u.add( nds[nidx].lvid );  // just lvid of subtree top, not the transform
    for(unsigned i=0 ; i < progeny.size() ; i++) u.add(digs[progeny[i]]) ; 
    return u.finalize() ; 
}

inline std::string stree::desc_node(int nidx) const 
{
    const snode& nd = nds[nidx]; 
    assert( nd.index == nidx ); 
    std::stringstream ss ; 

    ss << "stree:desc_node " 
       << nd.desc()
       //<< " " << subs[nidx] 
       << " " << soname[nd.lvid] 
       ;

    std::string s = ss.str(); 
    return s ; 
}

inline std::string stree::desc_nodes(const std::vector<int>& vnidx, unsigned edgeitems) const 
{
    std::stringstream ss ; 
    ss << "stree::desc_nodes " << vnidx.size() << std::endl ; 
    for(unsigned i=0 ; i < vnidx.size() ; i++) 
    {
        if( i < edgeitems || ( i > vnidx.size() - edgeitems )) 
            ss << desc_node(vnidx[i]) << std::endl ;  
        else if( i == edgeitems ) ss << " ... " << std::endl ;
    } 
    std::string s = ss.str(); 
    return s ; 
}

inline void stree::save( const char* fold ) const 
{
    NP::Write<int>(    fold, NDS, (int*)nds.data(),    nds.size(), snode::NV ); 
    NP::Write<double>( fold, TRS, (double*)trs.data(), trs.size(), 4, 4 ); 
    NP::WriteNames( fold, SONAME, soname ); 
    NP::WriteNames( fold, DIGS,   digs ); 
    NP::WriteNames( fold, SUBS,   subs ); 
    if(subs_freq) subs_freq->save(fold, SUBS_FREQ);  
}

inline void stree::load( const char* fold )
{
    NP* a_nds = NP::Load(fold, NDS); 
    nds.resize(a_nds->shape[0]); 
    memcpy( (int*)nds.data(),    a_nds->cvalues<int>() ,    a_nds->arr_bytes() ); 

    NP* a_trs = NP::Load(fold, TRS); 
    trs.resize(a_trs->shape[0]); 
    memcpy( (double*)trs.data(), a_trs->cvalues<double>() , a_trs->arr_bytes() ); 

    NP::ReadNames( fold, SONAME, soname ); 
    NP::ReadNames( fold, DIGS,   digs ); 
    NP::ReadNames( fold, SUBS,   subs ); 

    if(subs_freq) subs_freq->load(fold, SUBS_FREQ) ; 

}





struct U4Tree
{
    stree* st ; 
    const G4VPhysicalVolume* const top ; 
    std::map<const G4LogicalVolume* const, int> lvidx ;

    U4Tree(stree* st, const G4VPhysicalVolume* const top=nullptr ); 

    void init(); 

    void convertSolids(); 
    void convertSolids_r(const G4VPhysicalVolume* const pv); 
    void convertSolid(const G4LogicalVolume* const lv); 

    void convertNodes(); 
    int  convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, snode* parent ); 

}; 


U4Tree::U4Tree(stree* st_, const G4VPhysicalVolume* const top_)
    :
    st(st_),
    top(top_)
{
    init(); 
}

inline void U4Tree::init() 
{
    if(top == nullptr) return ; 
    convertSolids();
    convertNodes(); 
}

inline void U4Tree::convertSolids()
{
    convertSolids_r(top); 
}
inline void U4Tree::convertSolids_r(const G4VPhysicalVolume* const pv)
{
    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    int num_child = int(lv->GetNoDaughters()) ;  
    for (int i=0 ; i < num_child ;i++ ) convertSolids_r( lv->GetDaughter(i) ); 

    if(lvidx.find(lv) == lvidx.end()) convertSolid(lv); 
}

inline void U4Tree::convertSolid(const G4LogicalVolume* const lv)
{
    lvidx[lv] = lvidx.size(); 

    const G4VSolid* const solid = lv->GetSolid(); 
    G4String  soname_ = solid->GetName() ;   // returns by value, not reference
    st->soname.push_back(soname_); 
}

/**
U4Tree::convertNodes
-----------------------------

Serialize the n-ary tree into nds and trs vectors with 
structure info and transforms. 

**/

inline void U4Tree::convertNodes()
{
    convertNodes_r(top, 0, -1, nullptr ); 
}

inline int U4Tree::convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, snode* parent )
{
    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    int num_child = int(lv->GetNoDaughters()) ;  
    int lvid = lvidx[lv] ; 

    glm::tmat4x4<double> tr ;  
    U4Transform::GetObjectTransform(tr, pv); 
    st->trs.push_back(tr);  

    snode nd ; 
    nd.index = st->nds.size();
    nd.depth = depth ;   
    nd.sibdex = sibdex ; 
    nd.parent = parent ? parent->index : -1 ;  
    nd.num_child = num_child ; 
    nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    nd.next_sibling = -1 ; 
    nd.lvid = lvid ; 

    st->nds.push_back(nd); 

    std::string dig = stree::Digest(lvid, tr); 
    st->digs.push_back(dig); 

    if(sibdex == 0 && nd.parent > -1) st->nds[nd.parent].first_child = nd.index ; 
    // record first_child nidx into parent snode

    int p_sib = -1 ; 
    int i_sib = -1 ; 

    for (int i=0 ; i < num_child ;i++ ) 
    {
        p_sib = i_sib ; 
        i_sib = convertNodes_r( lv->GetDaughter(i), depth+1, i, &nd ); 
        if(p_sib > -1) st->nds[p_sib].next_sibling = i_sib ;    
        // sib->sib linkage, defallt -1 
    }
    return nd.index ; 
}


