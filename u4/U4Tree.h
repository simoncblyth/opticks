#pragma once
/**
U4Tree.h : explore minimal approach to geometry translation
==============================================================

* see also sysrap/stree.h sysrap/tests/stree_test.cc

**/


#include <map>
#include <algorithm>
#include <string>
#include <sstream>

#include <glm/glm.hpp>
#include "G4VPhysicalVolume.hh"

#include "NP.hh"
#include "sdigest.h"
#include "sfreq.h"
#include "stree.h"

#include "U4Transform.h"

struct U4Tree
{
    stree* st ; 
    const G4VPhysicalVolume* const top ; 
    std::map<const G4LogicalVolume* const, int> lvidx ;

    U4Tree(stree* st, const G4VPhysicalVolume* const top=nullptr ); 

    void convertSolids(); 
    void convertSolids_r(const G4VPhysicalVolume* const pv); 
    void convertSolid(const G4LogicalVolume* const lv); 

    void convertNodes(); 
    int  convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, snode* parent ); 
}; 


inline U4Tree::U4Tree(stree* st_, const G4VPhysicalVolume* const top_)
    :
    st(st_),
    top(top_)
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

Serialize the n-ary tree into nds and trs vectors within stree 
holding structural node info and transforms. 

**/

inline void U4Tree::convertNodes()
{
    convertNodes_r(top, 0, -1, nullptr ); 
}

/**
U4Tree::convertNodes_r
-----------------------

Most of the visit is preorder before the recursive call, 
but sibling to sibling links are done within the 
sibling loop using the node index returned by the 
recursive call. 

**/
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
        // sib->sib linkage, default -1 
    }
    return nd.index ; 
}

