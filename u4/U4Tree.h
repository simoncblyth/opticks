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
#include "G4PVPlacement.hh"

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
    std::vector<const G4VPhysicalVolume*> pvs ; 

    U4Tree(stree* st, const G4VPhysicalVolume* const top=nullptr ); 

    void convertSolids(); 
    void convertSolids_r(const G4VPhysicalVolume* const pv); 
    void convertSolid(const G4LogicalVolume* const lv); 

    void convertNodes(); 
    int  convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, snode* parent ); 

    const G4VPhysicalVolume* get_pv_(int nidx) const ; 
    const G4PVPlacement*     get_pv(int nidx) const ; 
    int                      get_pv_copyno(int nidx) const ; 

    int get_nidx(const G4VPhysicalVolume* pv) const ; 
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

    const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(pv) ;
    int copyno = pvp ? pvp->GetCopyNo() : -1 ;

    glm::tmat4x4<double> tr_m2w(1.) ;  
    U4Transform::GetObjectTransform(tr_m2w, pv); 

    glm::tmat4x4<double> tr_w2m(1.) ;  
    U4Transform::GetFrameTransform(tr_w2m, pv); 

    // HMM: the frame is inverse to obj transform sometimes : when no rotation 

    st->m2w.push_back(tr_m2w);  
    st->w2m.push_back(tr_w2m);  
    pvs.push_back(pv); 

    snode nd ; 

    nd.index = st->nds.size();
    nd.depth = depth ;   
    nd.sibdex = sibdex ; 
    nd.parent = parent ? parent->index : -1 ;  

    nd.num_child = num_child ; 
    nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    nd.next_sibling = -1 ; 
    nd.lvid = lvid ; 
    nd.copyno = copyno ; 

    st->nds.push_back(nd); 

    std::string dig = stree::Digest(lvid, tr_m2w); 
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


inline const G4VPhysicalVolume* U4Tree::get_pv_(int nidx) const 
{
    return nidx > -1 && nidx < int(pvs.size()) ? pvs[nidx] : nullptr ; 
}
inline const G4PVPlacement* U4Tree::get_pv(int nidx) const 
{
    const G4VPhysicalVolume* pv_ = get_pv_(nidx); 
    return dynamic_cast<const G4PVPlacement*>(pv_) ;
}
inline int U4Tree::get_pv_copyno(int nidx) const 
{
    const G4PVPlacement* pv = get_pv(nidx) ; 
    return pv ? pv->GetCopyNo() : -1 ; 
}


inline int U4Tree::get_nidx(const G4VPhysicalVolume* pv) const 
{
    int nidx = std::distance( pvs.begin(), std::find( pvs.begin(), pvs.end(), pv ) ) ;  
    return nidx < int(pvs.size()) ? nidx : -1 ;  
}











