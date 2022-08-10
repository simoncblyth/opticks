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
#include "G4Material.hh"

#include "NP.hh"
#include "sdigest.h"
#include "sfreq.h"
#include "stree.h"

#include "U4Transform.h"
#include "U4Material.hh"

struct U4Tree
{
    static U4Tree* Create( const G4VPhysicalVolume* const top ); 

    stree* st ; 
    const G4VPhysicalVolume* const top ; 
    int sensorCount ; 
    std::map<const G4LogicalVolume* const, int> lvidx ;
    std::vector<const G4VPhysicalVolume*> pvs ; 
    std::vector<const G4Material*>  materials ; 

    U4Tree(stree* st, const G4VPhysicalVolume* const top=nullptr ); 


    void convertMaterials(); 
    void convertMaterials_r(const G4VPhysicalVolume* const pv); 
    void convertMaterial(const G4Material* const mt); 

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


U4Tree* U4Tree::Create( const G4VPhysicalVolume* const top ) 
{
    std::cout << "[ U4Tree::Create " << std::endl ; 
    stree* st = new stree ; 
    U4Tree* tree = new U4Tree(st, top) ;
    st->factorize(); 
    std::cout << "] U4Tree::Create " << std::endl ; 
    return tree ; 
}


inline U4Tree::U4Tree(stree* st_, const G4VPhysicalVolume* const top_)
    :
    st(st_),
    top(top_),
    sensorCount(0)
{
    if(top == nullptr) return ; 
    convertMaterials();
    convertSolids();
    convertNodes(); 
}


/**
U4Tree::convertMaterials
-----------------------------------

1. recursive traverse collecting material pointers from all active LV into materials vector 
   in postorder of first encounter.

2. create NPFold mtfold holding properties of all active materials 

**/

inline void U4Tree::convertMaterials()
{
    convertMaterials_r(top); 
    st->mtfold = U4Material::MakePropertyFold(materials); 
}
inline void U4Tree::convertMaterials_r(const G4VPhysicalVolume* const pv)
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ ) convertMaterials_r( lv->GetDaughter(i) ); 
    G4Material* mt = lv->GetMaterial() ; 
    if(mt && (std::find(materials.begin(), materials.end(), mt) == materials.end())) convertMaterial(mt);  
}
inline void U4Tree::convertMaterial(const G4Material* const mt)
{
    materials.push_back(mt); 
    const G4String& mtname = mt->GetName() ;  
    st->mtname.push_back(mtname); 
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
    G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;  


    int num_child = int(lv->GetNoDaughters()) ;  
    int lvid = lvidx[lv] ; 

    const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(pv) ;
    int copyno = pvp ? pvp->GetCopyNo() : -1 ;

    glm::tmat4x4<double> tr_m2w(1.) ;  
    U4Transform::GetObjectTransform(tr_m2w, pv); 

    glm::tmat4x4<double> tr_w2m(1.) ;  
    U4Transform::GetFrameTransform(tr_w2m, pv); 


    st->m2w.push_back(tr_m2w);  
    st->w2m.push_back(tr_w2m);  
    pvs.push_back(pv); 


    int nidx = st->nds.size() ;
    //std::cout << "SD " << ( sd ? "Y" : "N" )  << " nidx " << std::setw(7) << nidx << " lvid " << std::setw(3) << lvid << std::endl ;  

    snode nd ; 

    nd.index = nidx ;
    nd.depth = depth ;   
    nd.sibdex = sibdex ; 
    nd.parent = parent ? parent->index : -1 ;  

    nd.num_child = num_child ; 
    nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    nd.next_sibling = -1 ; 
    nd.lvid = lvid ; 
    nd.copyno = copyno ; 
    nd.sensor = sd ? sensorCount : -1 ; 
    if(sd) sensorCount += 1 ; 


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











