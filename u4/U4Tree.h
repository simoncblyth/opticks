#pragma once
/**
U4Tree.h : explore minimal approach to geometry translation
==============================================================

Almost all the geometry information is populated 
and persisted within the *st* (sysrap/stree.h) member
The other members simply hold on to Geant4 pointers
that make no sense to persist and cannot live 
within sysrap as that does not depend on G4 headers.  

Actually header only impls can live anywhere, so 
U4Tree.h could move to sysrap as S4Tree.h.  
BUT: are more comfortable to do that only with small highly focussed headers. 

Currently this header is not so big, but expect that this will 
grow into the central header of the more direct geometry translation. 

See also:

* sysrap/stree.h 
* sysrap/tests/stree_test.cc

logging
---------

Cannot have U4Tree SLOG::EnvLevel because is currently header only, 
and cannot easily init a static in header only situation in C++11, 
With C++17 can supposedly do this easily with "inline static". See

https://stackoverflow.com/questions/11709859/how-to-have-static-data-members-in-a-header-only-library

Pragmatic workaround for runtime logging level is to 
adopt the log level int from SSim st.level which is 
controlled via envvar::

    export SSim__stree_level=1   # 0:one 1:minimal 2:some 3:verbose 4:extreme


**/


#include <map>
#include <algorithm>
#include <string>
#include <sstream>

#include <glm/glm.hpp>
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Material.hh"
#include "G4LogicalSurface.hh"
#include "G4OpRayleigh.hh"
#include "G4OpticalPhoton.hh"

#include "NP.hh"

#include "sdigest.h"
#include "sfreq.h"
#include "stree.h"
#include "suniquename.h"
#include "snd.hh"
#include "sdomain.h"
#include "ssys.h"

#include "SSimtrace.h"
#include "SEventConfig.hh"

#include "U4SensorIdentifier.h"
#include "U4SensorIdentifierDefault.h"

#include "U4Transform.h"
#include "U4Material.hh"
#include "U4Mat.h"

#include "U4Surface.h"
#include "U4SurfacePerfect.h"
#include "U4SurfaceArray.h"

#include "U4Scint.h"

#include "U4Solid.h"
#include "U4PhysicsTable.h"
#include "U4MaterialTable.h"
#include "U4TreeBorder.h"


struct U4Tree
{
    friend struct U4SimtraceTest ; // for U4Tree ctor  

    stree*                                      st ; 
    const G4VPhysicalVolume* const              top ; 
    U4SensorIdentifier*                         sid ; 
    int                                         level ;    
    // export SSim__stree_level=1 controls this 

    std::map<const G4LogicalVolume* const, int> lvidx ;
    std::vector<const G4VPhysicalVolume*>       pvs ; 
    std::vector<const G4Material*>              materials ; 
    std::vector<const G4LogicalSurface*>        surfaces ;   // both skin and border 
    int                                         num_surface_standard ;  // not including implicits
    std::vector<const G4VSolid*>                solids ; 
    U4PhysicsTable<G4OpRayleigh>*               rayleigh_table ; 
    U4Scint*                                    scint ;         

    // disable the below with settings with by defining the below envvar
    static constexpr const char* __DISABLE_OSUR_IMPLICIT = "U4Tree__DISABLE_OSUR_IMPLICIT" ; 
    static constexpr const char* __DISABLE_ISUR_IMPLICIT = "U4Tree__DISABLE_ISUR_IMPLICIT" ; 
    bool                                        enable_osur ; 
    bool                                        enable_isur ; 

    static U4Tree* Create( 
        stree* st, 
        const G4VPhysicalVolume* const top, 
        U4SensorIdentifier* sid=nullptr 
        ); 

    // using SSim::Get SSim::get_tree is tempting 
    // but that would add SSim dependency and SSim is not header only 
    // static U4Tree* Create( const G4VPhysicalVolume* const top, U4SensorIdentifier* sid=nullptr ); 

private:
    U4Tree(
        stree* st, 
        const G4VPhysicalVolume* const top=nullptr, 
        U4SensorIdentifier* sid=nullptr 
        ); 

    void init(); 

    static U4PhysicsTable<G4OpRayleigh>* CreateRayleighTable(); 
    void initRayleigh();   
    void initMaterials(); 
    void initMaterials_NoRINDEX(); 

    void initMaterials_r(const G4VPhysicalVolume* const pv); 
    void initMaterial(const G4Material* const mt); 

    void initScint(); 
    void initSurfaces(); 

    void initSolids(); 
    void initSolids_r(const G4VPhysicalVolume* const pv); 
    void initSolid(const G4LogicalVolume* const lv); 


    void initSolid(const G4VSolid* const so, int lvid ); 

    void initNodes(); 
    int  initNodes_r( 
        const G4VPhysicalVolume* const pv, 
        const G4VPhysicalVolume* const pv_p,
        int depth, 
        int sibdex, 
        int parent 
        ); 

    void initSurfaces_Serialize(); 
    void initStandard(); 

public:   //  accessors
    const G4Material*       getMaterial(int idx) const ; 
    const G4LogicalSurface* getSurface(int idx) const ; 

    std::string              desc() const ; 
    const G4VPhysicalVolume* get_pv_(int nidx) const ; 
    const G4PVPlacement*     get_pv(int nidx) const ; 
    int                      get_pv_copyno(int nidx) const ; 
    int get_nidx(const G4VPhysicalVolume* pv) const ; 

private:
    // identifySensitive called from U4Tree::Create
    void identifySensitive(); 
    void identifySensitiveInstances(); 
    void identifySensitiveGlobals(); 

public:
    void simtrace_scan(const char* base ) const ; 
    static void SimtraceScan( const G4VPhysicalVolume* const pv, const char* base ); 
}; 




/**
U4Tree::Create
----------------

Canonically invoked from G4CXOpticks::setGeometry

HMM: can these be moved into U4Tree ctor now ? 

**/
inline U4Tree* U4Tree::Create( 
    stree* st, 
    const G4VPhysicalVolume* const top, 
    U4SensorIdentifier* sid 
    ) 
{
    if(st->level > 0) std::cout << "[ U4Tree::Create " << std::endl ; 

    U4Tree* tree = new U4Tree(st, top, sid ) ;

    st->factorize(); 

    tree->identifySensitive(); 

    st->add_inst(); 

    if(st->level > 0) std::cout << "] U4Tree::Create " << std::endl ; 

    st->postcreate() ;  

    return tree ; 
}

inline U4Tree::U4Tree(
    stree* st_, 
    const G4VPhysicalVolume* const top_,  
    U4SensorIdentifier* sid_ 
    )
    :
    st(st_),
    top(top_),
    sid(sid_ ? sid_ : new U4SensorIdentifierDefault),
    level(st->level),
    num_surface_standard(-1),
    rayleigh_table(CreateRayleighTable()),
    scint(nullptr),
    enable_osur(!ssys::getenvbool(__DISABLE_OSUR_IMPLICIT)),
    enable_isur(!ssys::getenvbool(__DISABLE_ISUR_IMPLICIT))
{
    init(); 
}

/**
U4Tree::init
--------------

**/

inline void U4Tree::init()
{
    if(top == nullptr) return ; 

    initRayleigh(); 
    initMaterials();
    initMaterials_NoRINDEX(); 

    initScint();

    initSurfaces();
    initSolids();
    initNodes(); 
    initSurfaces_Serialize();

    initStandard(); 

    std::cerr << "U4Tree::init " <<  desc() << std::endl; 

}


/**
U4Tree::initMaterials
-----------------------

Canonically invoked from U4Tree::init 

1. recursive traverse collecting material pointers from all active 
   LV into materials vector in postorder of first encounter.

2. creates SSim/stree/material holding properties of all active materials

3. creates standard *mat* array using U4Material::MakeStandardArray 
   from the MPT of the materials, with an override for Water/RAYLEIGH
   from the rayleigh_table. The override is needed as G4OpRayleigh
   calculates RAYLEIGH scattering lengths from RINDEX for materials named
   "Water". 

NOTE THAT MATERIALS NAMED "vetoWater" ARE NOT SPECIAL CASED
SO THERE WILL BE MUCH LESS SCATTERING IN "vetoWater" THAN IN "Water"

**/

inline void U4Tree::initMaterials()
{
    initMaterials_r(top); 
    st->material = U4Material::MakePropertyFold(materials);  

    std::map<std::string, G4PhysicsVector*> prop_override ; 

    G4PhysicsVector* Water_RAYLEIGH = rayleigh_table->find("Water") ;  
    if(Water_RAYLEIGH) prop_override["Water/RAYLEIGH"] = Water_RAYLEIGH ; 

    st->standard->mat = U4Material::MakeStandardArray(materials, prop_override) ; 
}

inline void U4Tree::initMaterials_NoRINDEX()
{
    int num_materials = materials.size() ; 
    for(int i=0 ; i < num_materials ; i++)
    {
        const G4Material* mt = materials[i] ;    
        const char* mtn = mt->GetName().c_str(); 
        const G4MaterialPropertyVector* rindex = U4Mat::GetRINDEX( mt ) ; 
        if(rindex == nullptr) st->mtname_no_rindex.push_back(mtn) ; 
    }
}


/**
U4Tree::initScint
------------------

**/

inline void U4Tree::initScint()
{
    scint = U4Scint::Create(st->material) ; 
    if(scint) 
    {
        st->standard->icdf = scint->icdf ; 
    }
}

/**
U4Tree::CreateRayleighTable
----------------------------

Trying to find pre-existing G4OpRayleigh process
with the argumentless U4PhysicsTable ctor fails 
when U4Tree instanciation happens where it does currently.  
As a workaround pass in a throwaway G4OpRayleigh 
just to get access to its physics table. 

**/

inline U4PhysicsTable<G4OpRayleigh>* U4Tree::CreateRayleighTable() // static
{
    G4OpRayleigh* proc = new G4OpRayleigh ; 

    G4ParticleDefinition* OpticalPhoton = G4OpticalPhoton::Definition() ; 
    proc->BuildPhysicsTable(*OpticalPhoton); 

    U4PhysicsTable<G4OpRayleigh>* tab = new U4PhysicsTable<G4OpRayleigh>(proc) ; 
    return tab ; 
}


/**
U4Tree::initRayleigh
---------------------

Retain pointer from rayleigh_table formed in ctor into 
stree.standard.rayleigh

**/

inline void U4Tree::initRayleigh()
{
    if(level > 0) std::cerr 
        << "U4Tree::initRayleigh" 
        << " rayleigh_table " << std::endl  
        << ( rayleigh_table ? rayleigh_table->desc() : "-" ) 
        << std::endl 
        ;

    st->standard->rayleigh = rayleigh_table ? rayleigh_table->tab : nullptr  ; 
}


inline void U4Tree::initMaterials_r(const G4VPhysicalVolume* const pv)
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    int num_child = int(lv->GetNoDaughters()) ;  
    for (int i=0 ; i < num_child ;i++ ) initMaterials_r( lv->GetDaughter(i) ); 

    // postorder visit after recursive call  
    G4Material* mt = lv->GetMaterial() ; 
    assert(mt);  

    std::vector<const G4Material*>& m = materials ;  
    if(std::find(m.begin(), m.end(), mt) == m.end()) initMaterial(mt);  
}
inline void U4Tree::initMaterial(const G4Material* const mt)
{
    materials.push_back(mt); 
    const G4String& _mtname = mt->GetName() ;  
    unsigned g4index = mt->GetIndex() ;  
    const char* mtname = _mtname.c_str();   
    st->add_material( mtname, g4index  ); 
}


/**
U4Tree::initSurfaces
----------------------

1. U4Surface::Collect G4LogicalBorderSurface, G4LogicalSkinSurface pointers 
   into *surfaces* vector of G4LogicalSurface

2. Create stree::surface NPFold with U4Surface::MakeFold with all the 
   surface properties

2. collect surface indices and names into stree with with stree::add_surface 

**/


inline void U4Tree::initSurfaces()
{
    U4Surface::Collect(surfaces);  
    st->surface = U4Surface::MakeFold(surfaces); 
    num_surface_standard = int(surfaces.size()) ; 

    for(int i=0 ; i < num_surface_standard ; i++)
    {
        const G4LogicalSurface* ls = surfaces[i] ; 
        const G4String& name_ = ls->GetName() ; 
        const char* name = name_.c_str(); 
        st->add_surface(name);  
    }
}

/**
U4Tree::initSurfaces_Serialize
-------------------------------

As this requires to run after implicit surfaces are 
collected in initNodes it is too soon to do this 
within initSurfaces

Its too late to add implicit names here because 
they are needed by stree::get_boundary_name 

Regarding the perfects, expect will need to add them 
earlier too, once develop a test geometry that uses them. 
Moved perfect name collection before serialize
HMM: tis unclear where they should be added ?

**What are perfect surfaces used for ?**

Vaguely recall that the purpose of the named perfect 
surfaces is with simple CSGFoundry forged geometries 
that piggyback off other (usually full) geometries 
for their material and surface properties. 
Hence there is no ordering problem as entire geometries are 
loaded and reused for the piggyback. 

All that is needed is to plant the perfects for 
subsequent reuse within the test geomerty.  

**/

inline void U4Tree::initSurfaces_Serialize()
{
    std::vector<U4SurfacePerfect> perfect ; 
    U4SurfacePerfect::Get(perfect);  // perfect Detect,Absorb,Specular,Diffuse surfaces 

    int num_perfect = perfect.size(); 
    for(int i=0 ; i < num_perfect ; i++)
    {
        const U4SurfacePerfect& perf = perfect[i] ; 
        const char* name = perf.name.c_str() ; 
        st->add_surface( name );   
    }


    U4SurfaceArray serialize(surfaces, st->implicit, perfect) ;   
    st->standard->sur = serialize.sur ; 

}



/**
U4Tree::initSolids
-------------------

Uses postorder recursive traverse, ie the "visit" is in the 
tail after the recursive call, to match the traverse used 
by GDML, and hence giving the same "postorder" indices
for the solid lvIdx.

The entire volume tree is recursed, but only the 
first occurence of each LV solid gets converted 
(because they are all the same).
Done this way to have consistent lvIdx soIdx indexing with GDML.

cf X4PhysicalVolume::convertSolids 

**/

inline void U4Tree::initSolids()
{
    initSolids_r(top); 
}
inline void U4Tree::initSolids_r(const G4VPhysicalVolume* const pv)
{
    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
    int num_child = int(lv->GetNoDaughters()) ;  
    for (int i=0 ; i < num_child ;i++ ) initSolids_r( lv->GetDaughter(i) ); 

    // postorder visit after recursive call 
    if(lvidx.find(lv) == lvidx.end()) initSolid(lv); 
}
inline void U4Tree::initSolid(const G4LogicalVolume* const lv)
{
    int lvid = lvidx.size() ;  
    lvidx[lv] = lvid ; 
    const G4VSolid* const so = lv->GetSolid(); 
    initSolid(so, lvid); 
}






/**
U4Tree::initSolid
----------------------

Decided that intermediate CSG node tree is needed, 
as too difficult to leap direct from G4 to CSG 
models and a dependency fire break is advantageous. 

cf X4PhysicalVolume::ConvertSolid_ X4Solid::Convert


HMM: this could be the place to branch for 
special handling of deep CSG trees, based on 
hints planted in the G4VSolid name of the root
solid. Doing up here rather than within U4Solid::Convert
would avoid recursive complications. 

BUT: could rely on CSG_LISTNODE hints within the 
tree to direct the alt conversion 

**/

inline void U4Tree::initSolid(const G4VSolid* const so, int lvid )
{
    G4String _name = so->GetName() ; // bizarre: G4VSolid::GetName returns by value, not reference
    const char* name = _name.c_str();    

    assert( int(solids.size()) == lvid ); 
    int d = 0 ; 
#ifdef WITH_SND
    int root = U4Solid::Convert(so, lvid, d );  
    assert( root > -1 ); 
#else
    sn* root = U4Solid::Convert(so, lvid, d );  
    assert( root ); 
#endif

    solids.push_back(so);
    st->soname.push_back(name); 
    st->solids.push_back(root); 
} 



/**
U4Tree::initNodes
-----------------

Serialize the n-ary tree of structural nodes (the volumes) into nds and trs 
vectors within stree holding structural node info and transforms. 

Q: Is the surfaces vector complete before this runs ?
A: YES, U4Tree::initSurfaces collects the vector of surfaces before initNodes
   runs, so can rely on not meeting new standard surfaces in initNodes.  

**/

inline void U4Tree::initNodes()
{
    int nidx = initNodes_r(top, nullptr, 0, -1, -1 ); 
    assert( 0 == nidx ); 
}

/**
U4Tree::initNodes_r
-----------------------

Most of the visit is preorder before the recursive call, 
but sibling to sibling links are done within the 
sibling loop using the node index returned by the 
recursive call. 

The initial bd int4 (omat,osur,isur,imat) may have 
osur and isur overrides that add implicit surfaces when 
no prior surface is present and material properties are 
RINDEX->NoRINDEX. 

Implicit surfaces are needed for Opticks to reproduce Geant4 fStopAndKill 
behavior using additional perfect absorber surfaces in the Opticks 
geometry model that are not present in the Geant4 geometry model. 

From the Opticks point of view implicits and perfects are
just handled as standard surfaces with sur entries. 

enable_osur:false 
    reduces the number of implicits a lot, 
    which is convenient for initial testing

enable_osur:true
    THIS MUST BE ENABLED FOR Geant4 matching 
    WITHOUT IT SOME Water///Steel boundaries for example
    FAIL TO ABSORB PHOTONS : SEE ~/j/issues/3inch_PMT_geometry_after_virtual_delta.rst

**/



inline int U4Tree::initNodes_r( 
    const G4VPhysicalVolume* const pv, 
    const G4VPhysicalVolume* const pv_p, 
    int depth, 
    int sibdex, 
    int parent )
{
    // preorder visit before recursive call 
    U4TreeBorder border(st, pv, pv_p) ; 
   
    int omat = stree::GetPointerIndex<G4Material>(      materials, border.omat_); 
    int osur = stree::GetPointerIndex<G4LogicalSurface>(surfaces,  border.osur_); 
    int isur = stree::GetPointerIndex<G4LogicalSurface>(surfaces,  border.isur_); 
    int imat = stree::GetPointerIndex<G4Material>(      materials, border.imat_); 

    int4 bd = {omat, osur, isur, imat } ; 

    // overrides add implicit surfaces when no prior surface and RINDEX->NoRINDEX 
    if(enable_osur && border.has_osur_override(bd)) border.do_osur_override(bd);  
    if(enable_isur && border.has_isur_override(bd)) border.do_isur_override(bd); 

    border.check(bd); 
    int boundary = st->add_boundary(bd) ; 
    assert( boundary > -1 ); 


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
    nd.first_child = -1 ;  // gets changed inplace from lower recursion level 
    nd.next_sibling = -1 ; 
    nd.lvid = lvid ; 

    nd.copyno = copyno ; 
    nd.boundary = boundary ; 

    nd.sensor_id = -1 ;      // HMM: -1 to mean not-a-sensor is problematic GPU side
    nd.sensor_index = -1 ;  
    nd.sensor_name = -1 ; 
    // changed later by U4Tree::identifySensitiveInstances and stree::reorderSensors

    nd.repeat_index = 0 ;   
    nd.repeat_ordinal = -1 ;  
    // changed for instance subtrees by stree::labelFactorSubtrees, remainder left 0/-1 

    pvs.push_back(pv); 

    st->nds.push_back(nd); 
    st->digs.push_back(dig); 
    st->m2w.push_back(tr_m2w);  
    st->w2m.push_back(tr_w2m);  


    // "GGeo Transform Debug" comparison
    glm::tmat4x4<double> tt_gtd(1.) ;   
    glm::tmat4x4<double> vv_gtd(1.) ;

    bool local = false ; 
    bool reverse = false ; 
    st->get_node_product( tt_gtd, vv_gtd, nidx, local, reverse, nullptr );   
    // product of m2w transforms from root down to nidx,  
    // must be after push_backs of nd and tr_m2w

    st->gtd.push_back(tt_gtd);  


    if(sibdex == 0 && nd.parent > -1) st->nds[nd.parent].first_child = nd.index ; 
    // record first_child nidx into parent snode by reaching up thru the recursion levels 

    int p_sib = -1 ; 
    int i_sib = -1 ; 
    for (int i=0 ; i < num_child ;i++ ) 
    {
        p_sib = i_sib ;    
        // node index of previous child gets set for i > 0

        //                    ch_pv ch_parent_pv ch_depth ch_sibdex ch_parent    
        i_sib = initNodes_r( lv->GetDaughter(i), pv, depth+1, i, nd.index ); 

        if(p_sib > -1) st->nds[p_sib].next_sibling = i_sib ; 
        // after first child, reach back to previous sibling snode 
        // to set the sib->sib linkage, default -1
    }

    return nd.index ; 
}






/**
U4Tree::initStandard
----------------------

Have now transitioned from a former unholy mixture of old and new.
But still in validation stage, so retain the below old notes for now. 

SSim::import_bnd
GGeo::convertSim_BndLib

X4PhysicalVolume::addBoundary
   during volume traversal look for border/skin surfaces and
   adds to GBndLib when found using the names of boundaries and surfaces 

   * uses GGeo::findSkinSurface instead just use G4 (follow G4OpBoundaryProcess) 

float/double bnd buffer is a zip from GSurfaceLib GMaterialLib which 
uses domain standardization and is steered by the int bd buffer 

GMaterialLib/GSurfaceLib
   standard domain and standard set of props with defaults   

   * current 

How to handle the bnd cleanly ? 

SSim and SBnd look all setup, but they are currently relying on NP* bnd
with "spec" names that comes from GGeo/GBndLib 

So just need to create the bnd and the optical buffer ? 

**/

inline void U4Tree::initStandard()
{
    st->initStandard(); 
}






inline const G4Material* U4Tree::getMaterial(int idx) const
{
    return idx > -1 ? materials[idx] : nullptr ; 
}
inline const G4LogicalSurface* U4Tree::getSurface(int idx) const
{
    return idx > -1 ? surfaces[idx] : nullptr ; 
}

inline std::string U4Tree::desc() const 
{
    std::stringstream ss ; 
    ss << "U4Tree::desc" << std::endl 
       << " st "  << ( st ? "Y" : "N" ) << std::endl 
       << " top " << ( top ? "Y" : "N" ) << std::endl 
       << " sid " << ( sid ? "Y" : "N" ) << std::endl 
       << " level " << level << std::endl 
       << " lvidx " << lvidx.size() << std::endl
       << " pvs " << pvs.size() << std::endl 
       << " materials " << materials.size() << std::endl 
       << " surfaces " << surfaces.size() << std::endl 
       << " solids " << solids.size() << std::endl 
       << " enable_osur " << ( enable_osur ? "YES" : "NO " ) << std::endl 
       << " enable_isur " << ( enable_isur ? "YES" : "NO " ) << std::endl 
       ;  
    std::string str = ss.str(); 
    return str ; 
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

This is called from U4Tree::Create after U4Tree instanciation
and stree::factorize is called, but before stree::add_inst. 

Initially tried to simply use lv->GetSensitiveDetector() to 
identify sensor nodes by that is problematic because 
the SD is not on the volume with the copyNo and this 
use of copyNo is detector specific.  Also not all JUNO SD
are actually sensitive. 


1. identifySensitiveInstances : sets stree/snode sensor fields 
   of instance outer volume nodes

2. identifySensitiveGlobals : sets stree/snode sensor field
   of remainder nodes identified as sensitive 
   (none expected when using U4SensorIdentifierDefault) 

3. stree::reorderSensors

   * recursive nd traverse setting nd.sensor_index
   * nd loop collecting nd.sensor_id to update stree::sensor_id 


**/

inline void U4Tree::identifySensitive()
{
    if(level > 0) std::cerr 
        << "[ U4Tree::identifySensitive " 
        << std::endl 
        ; 

    st->sensor_count = 0 ; 

    identifySensitiveInstances();  
    identifySensitiveGlobals(); 

    st->reorderSensors(); 
    // change nd.sensor_index to facilitate comparison with GGeo


    if(level > 0) std::cerr 
        << "] U4Tree::identifySensitive"
        << " st.sensor_count " << st->sensor_count 
        << std::endl
        ; 
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

NB changes made to U4Tree::identifySensitiveInstances should
usually be made in tandem with U4Tree::identifySensitiveGlobals

**/

inline void U4Tree::identifySensitiveInstances()
{
    unsigned num_factor = st->get_num_factor(); 
    if(level > 0) std::cerr 
        << "[ U4Tree::identifySensitiveInstances" 
        << " num_factor " << num_factor 
        << " st.sensor_count " << st->sensor_count 
        << std::endl
        ; 

    for(unsigned i=0 ; i < num_factor ; i++)
    {
        std::vector<int> outer ; 
        st->get_factor_nodes(outer, i);  
        // nidx of outer volumes of the instances for each factor
 
        sfactor& fac = st->get_factor_(i); 
        fac.sensors = 0  ; 

        for(unsigned j=0 ; j < outer.size() ; j++)
        {
            int nidx = outer[j] ; 
            const G4VPhysicalVolume* pv = get_pv_(nidx) ; 
            const char* pvn = pv->GetName().c_str() ; 

            int sensor_id = sid->getInstanceIdentity(pv) ;  
            assert( sensor_id >= -1 );  // sensor_id:-1 signifies "not-a-sensor"

            int sensor_index = sensor_id > -1 ? st->sensor_count : -1 ; 
            int sensor_name = -1 ; 

            if(sensor_id > -1 ) 
            {
                st->sensor_count += 1 ;  // count over all factors  
                fac.sensors += 1 ;       // count sensors for each factor  
                sensor_name = suniquename::Add(pvn, st->sensor_name ) ; 
            }

            snode& nd = st->nds[nidx] ; 
            nd.sensor_id = sensor_id ; 
            nd.sensor_index = sensor_index ; 
            nd.sensor_name = sensor_name ; 

            if(level > 1) std::cerr
                << "U4Tree::identifySensitiveInstances"
                << " i " << std::setw(7) << i 
                << " sensor_id " << std::setw(7) << sensor_id  
                << " sensor_index " << std::setw(7) << sensor_index  
                << std::endl
                ;
        }

        if(level > 0) std::cerr 
            << "U4Tree::identifySensitiveInstances"
            << " factor " << i 
            << " fac.sensors " << fac.sensors
            << std::endl 
            ;   
    }

    if(level > 0) std::cerr 
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
    st->get_remainder_nidx(remainder) ;   

    if(level > 0) std::cerr 
        << "[ U4Tree::identifySensitiveGlobals" 
        << " st.sensor_count " << st->sensor_count 
        << " remainder.size " << remainder.size()
        << std::endl
        ; 

    for(unsigned i=0 ; i < remainder.size() ; i++)
    { 
        int nidx = remainder[i] ; 
        snode& nd = st->nds[nidx] ; 

        const G4VPhysicalVolume* pv = get_pv_(nidx) ; 
        const G4VPhysicalVolume* ppv = get_pv_(nd.parent) ; 

        const char* pvn = pv->GetName().c_str() ; 
        const char* ppvn = ppv ? ppv->GetName().c_str() : nullptr  ; 

        int sensor_id = sid->getGlobalIdentity(pv, ppv ) ;  
        int sensor_index = sensor_id > -1 ? st->sensor_count : -1 ; 
        int sensor_name = -1 ; 

        if(sensor_id > -1 ) 
        {
            st->sensor_count += 1 ;  // count over all factors  
            sensor_name = suniquename::Add(pvn, st->sensor_name ) ; 
        }
        nd.sensor_id = sensor_id ; 
        nd.sensor_index = sensor_index ; 
        nd.sensor_name = sensor_name ; 
 
        if(level > 1) std::cerr
            << "U4Tree::identifySensitiveGlobals"
            << " i " << std::setw(7) << i 
            << " nidx " << std::setw(6) << nidx
            << " sensor_id " << std::setw(7) << sensor_id  
            << " sensor_index " << std::setw(7) << sensor_index  
            << " pvn " << pvn 
            << " ppvn " << ( ppvn ? ppvn : "-" )
            << std::endl
            ;
    }
    if(level > 0) std::cerr 
        << "] U4Tree::identifySensitiveGlobals " 
        << " st.sensor_count " << st->sensor_count 
        << " remainder.size " << remainder.size()
        << std::endl 
        ; 
}




/**
U4Tree::simtrace_scan
------------------------

HMM: could use NPFold instead of saving to file after the scan of each solid ?
(actually the simple approach probably better in terms of memory)

Note that SEventConfig::RGModeLabel with U4SimtraceTest.sh 
starts as simulate (the default) and then gets changed to 
simtrace after the first SSimtrace::Scan call. 

**/

inline void U4Tree::simtrace_scan(const char* base ) const 
{
    LOG(info) << "[ " << SEventConfig::RGModeLabel() ; 
    st->save_trs(base); 
    assert( st->soname.size() == solids.size() );  
    for(unsigned i=0 ; i < st->soname.size() ; i++)  // over unique solid names
    {   
        const char* soname = st->soname[i].c_str(); 
        const G4VSolid* solid = solids[i] ; 
        G4String name = solid->GetName();  // bizarre by value 
        assert( strcmp( name.c_str(), soname ) == 0 );  

        LOG(info) << " i " << std::setw(3) << i << " RGMode: " << SEventConfig::RGModeLabel() ; 
        SSimtrace::Scan(solid, base) ;   
    } 
    LOG(info) << "] " << SEventConfig::RGModeLabel() ; 
}



/**
U4Tree::SimtraceScan
-----------------------

**/

inline void U4Tree::SimtraceScan( const G4VPhysicalVolume* const pv, const char* base ) // static
{
    stree st ; 
    U4Tree ut(&st, pv ) ; 
    ut.simtrace_scan(base) ; 
}


