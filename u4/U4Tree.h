#pragma once
/**
U4Tree.h : explore minimal approach to geometry translation
==============================================================

* note how persisting is mostly delegated to stree.h 
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

#include "U4SensorIdentifier.h"
#include "U4SensorIdentifierDefault.h"

#include "U4Transform.h"
#include "U4Material.hh"

struct U4Tree
{
    static constexpr const bool VERBOSE = true ; 
    static U4Tree* Create( stree* st, const G4VPhysicalVolume* const top, const U4SensorIdentifier* sid=nullptr ); 


    stree* st ; 
    const G4VPhysicalVolume* const top ; 
    const U4SensorIdentifier* sid ; 

    std::map<const G4LogicalVolume* const, int> lvidx ;
    std::vector<const G4VPhysicalVolume*> pvs ; 
    std::vector<const G4Material*>  materials ; 


    // HMM: should really be SSim argument now ?
    U4Tree(stree* st, const G4VPhysicalVolume* const top=nullptr, const U4SensorIdentifier* sid=nullptr ); 
    void init(); 

    void convertMaterials(); 
    void convertMaterials_r(const G4VPhysicalVolume* const pv); 
    void convertMaterial(const G4Material* const mt); 

    void convertSolids(); 
    void convertSolids_r(const G4VPhysicalVolume* const pv); 
    void convertSolid(const G4LogicalVolume* const lv); 

    void convertNodes(); 
    int  convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, int parent ); 

    const G4VPhysicalVolume* get_pv_(int nidx) const ; 
    const G4PVPlacement*     get_pv(int nidx) const ; 
    int                      get_pv_copyno(int nidx) const ; 

    int get_nidx(const G4VPhysicalVolume* pv) const ; 

    void identifySensitive(); 
    void identifySensitiveInstances(); 
    void identifySensitiveGlobals(); 

}; 




/**
U4Tree::Create
----------------

Canonically invoked from G4CXOpticks::setGeometry

**/
inline U4Tree* U4Tree::Create( stree* st, const G4VPhysicalVolume* const top, const U4SensorIdentifier* sid ) 
{
    if(VERBOSE) std::cout << "[ U4Tree::Create " << std::endl ; 
    U4Tree* tr = new U4Tree(st, top, sid ) ;

    st->factorize(); 
    tr->identifySensitive(); 
    st->add_inst(); 

    if(VERBOSE) std::cout << "] U4Tree::Create " << std::endl ; 
    return tr ; 
}


inline U4Tree::U4Tree(stree* st_, const G4VPhysicalVolume* const top_,  const U4SensorIdentifier* sid_ )
    :
    st(st_),
    top(top_),
    sid(sid_ ? sid_ : new U4SensorIdentifierDefault)
{
    init(); 
}

inline void U4Tree::init()
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

2. create SSim/stree/mtfold holding properties of all active materials 

   TODO: relocate to SSim/mtfold, rather than SSim/stree/mtfold ? 

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
    unsigned g4index = mt->GetIndex() ;  
    st->add_material( mtname, g4index  ); 
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
    convertNodes_r(top, 0, -1, -1 ); 
}

/**
U4Tree::convertNodes_r
-----------------------

Most of the visit is preorder before the recursive call, 
but sibling to sibling links are done within the 
sibling loop using the node index returned by the 
recursive call. 

Initially tried to simply use lv->GetSensitiveDetector() to 
identify sensor nodes by that is problematic because 
the SD is not on the volume with the copyNo and this 
use of copyNo is detector specific.  Also not all JUNO SD
are actually sensitive. 

**/
inline int U4Tree::convertNodes_r( const G4VPhysicalVolume* const pv, int depth, int sibdex, int parent )
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

    std::string dig = stree::Digest(lvid, tr_m2w); 

    int nidx = st->nds.size() ;  // 0-based node index

    snode nd ; 

    nd.index = nidx ;
    nd.depth = depth ;   
    nd.sibdex = sibdex ; 
    nd.parent = parent ;  

    nd.num_child = num_child ; 
    nd.first_child = -1 ;     // gets changed inplace from lower recursion level 
    nd.next_sibling = -1 ; 
    nd.lvid = lvid ; 
    nd.copyno = copyno ; 

    nd.sensor_id = -1 ;     // changed later by U4Tree::identifySensitiveInstances
    nd.sensor_index = -1 ;  // changed later by U4Tree::identifySensitiveInstances and stree::reorderSensors
    nd.repeat_index = 0 ;   // changed later for instance subtrees by stree::labelFactorSubtrees leaving remainder at 0 

    pvs.push_back(pv); 
    st->nds.push_back(nd); 
    st->digs.push_back(dig); 
    st->m2w.push_back(tr_m2w);  
    st->w2m.push_back(tr_w2m);  


    glm::tmat4x4<double> tr_gtd(1.) ;          // "GGeo Transform Debug" comparison
    st->get_m2w_product(tr_gtd, nidx, false );  // NB this must be after push back of nd and tr_m2w
    st->gtd.push_back(tr_gtd);  



    if(sibdex == 0 && nd.parent > -1) st->nds[nd.parent].first_child = nd.index ; 
    // record first_child nidx into parent snode

    int p_sib = -1 ; 
    int i_sib = -1 ; 
    for (int i=0 ; i < num_child ;i++ ) 
    {
        p_sib = i_sib ;  // node index of previous child 
        i_sib = convertNodes_r( lv->GetDaughter(i), depth+1, i, nd.index ); 
        if(p_sib > -1) st->nds[p_sib].next_sibling = i_sib ; 
    }
    // within the above loop, reach back to previous sibling snode to set the sib->sib linkage, default -1

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


/**
U4Tree::identifySensitive
----------------------------

**/

inline void U4Tree::identifySensitive()
{
    if(VERBOSE) std::cerr << "[ U4Tree::identifySensitive " << std::endl ; 
    st->sensor_count = 0 ; 

    identifySensitiveInstances(); 
    identifySensitiveGlobals(); 

    st->reorderSensors();  // change nd.sensor_index to facilitate comparison with GGeo
    if(VERBOSE) std::cerr << "] U4Tree::identifySensitive " << std::endl ; 
}



/**
U4Tree::identifySensitiveInstances
------------------------------------

Canonically invoked from U4Tree::Create after stree factorization
and before instance creation. 

This uses stree/sfactor to get node indices of the outer
volumes of all instances. These together with U4SensorIdentifier/sid
allow sensor_id and sensor_index results (-1 when not sensors) 
to be added into the stree/snode. 

These values are subsequently used by instance creation and 
are inserted into the instance transform fourth column. 

NOTE that the nd.sensor_index may be subsequently changed by 
stree::reorderSensors

TODO: 

This is assuming a full geometry with instances found, 
but what about a geometry where nothing got instanced and 
everything is in the remainder. Or a geometry with some sensors 
in remainder and some in factors

Need a way to traverse the tree from the root and skip the 
factored subtrees : easy way to do that is to label the tree with the ridx. 


**/

inline void U4Tree::identifySensitiveInstances()
{
    unsigned num_factor = st->get_num_factor(); 
    if(VERBOSE) std::cerr 
        << "[ U4Tree::identifySensitiveInstances" 
        << " num_factor " << num_factor 
        << " st.sensor_count " << st->sensor_count 
        << std::endl
        ; 

    for(unsigned i=0 ; i < num_factor ; i++)
    {
        std::vector<int> outer ; 
        st->get_factor_nodes(outer, i );  // nidx of outer volumes of instances 
        sfactor& fac = st->get_factor(i); 
        fac.sensors = 0  ; 

        for(unsigned j=0 ; j < outer.size() ; j++)
        {
            int nidx = outer[j] ; 
            const G4VPhysicalVolume* pv = get_pv_(nidx) ; 
            int sensor_id = sid->getInstanceIdentity(pv) ;  
            int sensor_index = sensor_id > -1 ? st->sensor_count : -1 ; 
            if(sensor_id > -1 ) 
            {
                st->sensor_count += 1 ;  // count over all factors  
                fac.sensors += 1 ;   // count sensors for each factor  
            }
            snode& nd = st->nds[nidx] ; 
            nd.sensor_id = sensor_id ; 
            nd.sensor_index = sensor_index ; 
        }
    }

    if(VERBOSE) std::cerr 
        << "] U4Tree::identifySensitiveInstances"
        << " num_factor " << num_factor
        << " st.sensor_count " << st->sensor_count 
        << std::endl 
        ; 
}

/**
U4Tree::identifySensitiveGlobals
----------------------------------

This remains rather untested as JUNO geometry does not have sensitive globals. 

**/

inline void U4Tree::identifySensitiveGlobals()
{
    std::vector<int> remainder ; 
    st->get_remainder_nodes(remainder) ;   

    if(VERBOSE) std::cerr 
        << "[ U4Tree::identifySensitiveGlobals" 
        << " st.sensor_count " << st->sensor_count 
        << " remainder.size " << remainder.size()
        << std::endl
        ; 

    for(unsigned i=0 ; i < remainder.size() ; i++)
    { 
        int nidx = remainder[i] ; 
        const G4VPhysicalVolume* pv = get_pv_(nidx) ; 
        int sensor_id = sid->getGlobalIdentity(pv) ;  
        int sensor_index = sensor_id > -1 ? st->sensor_count : -1 ; 

        if(sensor_id > -1 ) 
        {
            st->sensor_count += 1 ;  // count over all factors  
        }
        snode& nd = st->nds[nidx] ; 
        nd.sensor_id = sensor_id ; 
        nd.sensor_index = sensor_index ; 
    }
    if(VERBOSE) std::cerr 
        << "] U4Tree::identifySensitiveGlobals " 
        << " st.sensor_count " << st->sensor_count 
        << " remainder.size " << remainder.size()
        << std::endl 
        ; 
}




