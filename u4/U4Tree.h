#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <sstream>

#include <glm/glm.hpp>

#include "sdigest.h"
#include "G4VPhysicalVolume.hh"
#include "U4Transform.h"
#include "NP.hh"


class G4VPhysicalVolume ; 

struct Nd
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
}; 

struct U4Tree
{
    static constexpr const char* NDS = "nds.npy" ; 
    static constexpr const char* TRS = "trs.npy" ; 
    static constexpr const char* SONAME = "soname.txt" ; 
    static constexpr const char* DIGS = "digs.txt" ; 
    static constexpr const char* SUBS = "subs.txt" ; 

    const G4VPhysicalVolume* const top ; 
    std::map<const G4LogicalVolume* const, int> lvidx ;
    std::vector<std::string> soname ; 

    std::vector<glm::tmat4x4<double>> trs ; 
    std::vector<Nd> nds ; 
    std::vector<std::string> digs ; // single node digest  
    std::vector<std::string> subs ; // subtree digest 

    U4Tree(const G4VPhysicalVolume* const top=nullptr ); 

    void init(); 
    std::string desc() const ; 

    void convertSolids(); 
    void convertSolids_r(const G4VPhysicalVolume* const pv); 
    void convertSolid(const G4LogicalVolume* const lv); 

    void convertNodes(); 
    int  convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, Nd* parent ); 

    static std::string NodeDigest(int lvid, const glm::tmat4x4<double>& tr ); 
    static std::string NodeDigest(int lvid );

    void get_children(std::vector<int>& children, int nidx) const ; 
    void get_progeny_r( std::vector<int>& progeny, int nidx ) const ; 

    void classifySubtrees(); 
    std::string subtree_digest( int nidx );  

    void save( const char* fold ); 
    void load( const char* fold ); 
}; 


U4Tree::U4Tree(const G4VPhysicalVolume* const top_)
    :
    top(top_)
{
    init(); 
}


inline void U4Tree::init() 
{
    if(top == nullptr) return ; 
    convertSolids();
    convertNodes(); 
    classifySubtrees(); 
}


inline std::string U4Tree::desc() const 
{
    std::stringstream ss ; 
    ss << "U4Tree::desc"
       << " nds " << nds.size()
       << " trs " << trs.size()
       << " digs " << digs.size()
       << " subs " << subs.size()
       << " soname " << soname.size()
       ;
    std::string s = ss.str(); 
    return s ; 
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
    soname.push_back(soname_); 
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

inline int U4Tree::convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, Nd* parent )
{
    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    int num_child = int(lv->GetNoDaughters()) ;  
    int lvid = lvidx[lv] ; 

    glm::tmat4x4<double> tr ;  
    U4Transform::GetObjectTransform(tr, pv); 
    trs.push_back(tr);  

    Nd nd ; 

    nd.index = nds.size();
    nd.depth = depth ;   
    nd.sibdex = sibdex ; 
    nd.parent = parent ? parent->index : -1 ;  
    nd.num_child = num_child ; 
    nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    nd.next_sibling = -1 ; 
    nd.lvid = lvid ; 
    nds.push_back(nd); 

    std::string dig = NodeDigest(lvid, tr); 
    digs.push_back(dig); 
   

    if(sibdex == 0 && nd.parent > -1) nds[nd.parent].first_child = nd.index ; // change upper level 

    int p_sib = -1 ; 
    int i_sib = -1 ; 

    for (int i=0 ; i < num_child ;i++ ) 
    {
        p_sib = i_sib ; 
        i_sib = convertNodes_r( lv->GetDaughter(i), depth+1, i, &nd ); 
        if(p_sib > -1) nds[p_sib].next_sibling = i_sib ;    // sib->sib linkage, defallt -1 
    }
    return nd.index ; 
}


/**
U4Tree::node_digest
----------------------

Progeny digest needs to ncompassing transforms + lvid of subnodes, but only lvid of 
the node in question ?  

**/

inline std::string U4Tree::NodeDigest(int lvid, const glm::tmat4x4<double>& tr )
{
    sdigest u ; 
    u.add( lvid ); 
    u.add( (char*)glm::value_ptr(tr), sizeof(double)*16 ) ; 
    std::string dig = u.finalize(); 
    return dig ; 
} 

inline std::string U4Tree::NodeDigest(int lvid )
{
    return sdigest::Int(lvid); 
}

inline void U4Tree::get_children( std::vector<int>& children , int nidx ) const 
{
    const Nd& nd = nds[nidx]; 
    assert( nd.index == nidx ); 

    int ch = nd.first_child ; 
    while( ch > -1 )
    {
        const Nd& child = nds[ch] ; 
        assert( child.parent == nd.index ); 
        children.push_back(child.index); 
        ch = child.next_sibling ; 
    }
    assert( int(children.size()) == nd.num_child ); 
}


inline void U4Tree::get_progeny_r( std::vector<int>& progeny , int nidx ) const 
{
    std::vector<int> children ; 
    get_children(children, nidx); 
    std::copy(children.begin(), children.end(), std::back_inserter(progeny)); 
    for(unsigned i=0 ; i < children.size() ; i++) get_progeny_r(progeny, children[i] );
}

inline void U4Tree::classifySubtrees()
{
    std::cout << "[ U4Tree::classifySubtrees " << std::endl ; 
    for(int nidx=0 ; nidx < int(nds.size()) ; nidx++) 
        subs.push_back(subtree_digest(nidx)) ; 
    std::cout << "] U4Tree::classifySubtrees " << std::endl ; 
}

inline std::string U4Tree::subtree_digest( int nidx )
{
    std::vector<int> progeny ; 
    get_progeny_r(progeny, nidx); 

    sdigest u ;  
    for(unsigned i=0 ; i < progeny.size() ; i++) u.add(digs[progeny[i]]) ; 
    return u.finalize() ; 
}


inline void U4Tree::save( const char* fold )
{
    NP::Write<int>(    fold, NDS, (int*)nds.data(),    nds.size(), Nd::NV ); 
    NP::Write<double>( fold, TRS, (double*)trs.data(), trs.size(), 4, 4 ); 
    NP::WriteNames( fold, SONAME, soname ); 
    NP::WriteNames( fold, DIGS,   digs ); 
    NP::WriteNames( fold, SUBS,   subs ); 
}

inline void U4Tree::load( const char* fold )
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
}

