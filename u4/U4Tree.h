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
#include "snd.hh"
#include "sdomain.h"

#include "SSimtrace.h"
#include "SEventConfig.hh"

#include "U4SensorIdentifier.h"
#include "U4SensorIdentifierDefault.h"

#include "U4Transform.h"
#include "U4Material.hh"

#include "U4Surface.h"
#include "U4SurfacePerfect.h"
#include "U4SurfaceArray.h"

#include "U4Solid.h"
#include "U4PhysicsTable.h"
#include "U4MaterialTable.h"

/*
HMM: cannot have U4Tree EnvLevel because it is currently header only, 
and cannot easily init a static in header only situation in C++11, 
With C++17 can supposedly do this easily with "inline static". See

https://stackoverflow.com/questions/11709859/how-to-have-static-data-members-in-a-header-only-library

Pragmatic workaround for runtime logging level is to 
adopt the log level int from stree st.level which is 
controlled via envvar::

    export SSim__stree_level=1   # 0:one 1:minimal 2:some 3:verbose 4:extreme

*/

struct U4Tree
{
    stree* st ; 
    const G4VPhysicalVolume* const top ; 
    const U4SensorIdentifier* sid ; 
    int level ;    // use export SSim__stree_level=1  will control this 

    std::map<const G4LogicalVolume* const, int> lvidx ;
    std::vector<const G4VPhysicalVolume*> pvs ; 
    std::vector<const G4Material*>  materials ; 
    std::vector<const G4LogicalSurface*> surfaces ;   // both skin and border 
    int num_surfaces ; 

    std::vector<const G4VSolid*>    solids ; 
    U4PhysicsTable<G4OpRayleigh>* rayleigh_table ; 


    static const G4MaterialPropertyVector* GetRINDEX(   const G4Material* mt ); 
    static const G4MaterialPropertyVector* GetProperty( const G4Material* mt, int index ); 

    template<typename T>
    static int GetPointerIndex( const std::vector<const T*>& vec, const T* obj) ; 
    template<typename T>
    static int GetValueIndex( const std::vector<T>& vec, const T& obj) ; 


    // HMM: should really be SSim argument now ?
    static U4Tree* Create( stree* st, const G4VPhysicalVolume* const top, const U4SensorIdentifier* sid=nullptr ); 
    U4Tree(stree* st, const G4VPhysicalVolume* const top=nullptr, const U4SensorIdentifier* sid=nullptr ); 
    void init(); 
    void initDomain(); 

    static U4PhysicsTable<G4OpRayleigh>* CreateRayleighTable(); 
    void initRayleigh();   
    void initMaterials(); 
    void initMaterials_NoRINDEX(); 

    void initMaterials_r(const G4VPhysicalVolume* const pv); 
    void initMaterial(const G4Material* const mt); 

    void initSurfaces(); 

    void initSolids(); 
    void initSolids_r(const G4VPhysicalVolume* const pv); 
    void initSolid(const G4LogicalVolume* const lv); 
    void initSolid(const G4VSolid* const so, int lvid ); 


    void initNodes(); 
    int  initNodes_r( const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p, int depth, int sibdex, int parent ); 

    void initSurfaces_Serialize(); 
    void initBoundary();  // EMPTY IMPL

    //  accessors

    const G4Material*       getMaterial(int idx) const ; 
    const G4LogicalSurface* getSurface(int idx) const ; 

    std::string              desc() const ; 
    const G4VPhysicalVolume* get_pv_(int nidx) const ; 
    const G4PVPlacement*     get_pv(int nidx) const ; 
    int                      get_pv_copyno(int nidx) const ; 
    int get_nidx(const G4VPhysicalVolume* pv) const ; 

    // identify
    void identifySensitive(); 
    void identifySensitiveInstances(); 
    void identifySensitiveGlobals(); 

    void simtrace_scan(const char* base ) const ; 
    static void SimtraceScan( const G4VPhysicalVolume* const pv, const char* base ); 


}; 


inline const G4MaterialPropertyVector* U4Tree::GetRINDEX(  const G4Material* mt ) // static
{
    return GetProperty(mt, kRINDEX ); 
}
inline const G4MaterialPropertyVector* U4Tree::GetProperty(const G4Material* mt, int index ) // static
{
    G4MaterialPropertiesTable* mpt = mt ? mt->GetMaterialPropertiesTable() : nullptr ;
    const G4MaterialPropertyVector* mpv = mpt ? mpt->GetProperty(index) : nullptr ;    
    return mpv ; 
}



template<typename T>
inline int U4Tree::GetPointerIndex( const std::vector<const T*>& vec, const T* obj) // static
{
    if( obj == nullptr || vec.size() == 0 ) return -1 ; 
    size_t idx = std::distance( vec.begin(), std::find(vec.begin(), vec.end(), obj )); 
    return idx < vec.size() ? int(idx) : -1 ;   
}

template<typename T>
inline int U4Tree::GetValueIndex( const std::vector<T>& vec, const T& obj) // static 
{
    size_t idx = std::distance( vec.begin(), std::find(vec.begin(), vec.end(), obj )); 
    return idx < vec.size() ? int(idx) : -1 ;   
}


/**
U4Tree::Create
----------------

Canonically invoked from G4CXOpticks::setGeometry

**/
inline U4Tree* U4Tree::Create( stree* st, const G4VPhysicalVolume* const top, const U4SensorIdentifier* sid ) 
{
    if(st->level > 0) std::cout << "[ U4Tree::Create " << std::endl ; 
    U4Tree* tr = new U4Tree(st, top, sid ) ;

    st->factorize(); 
    tr->identifySensitive(); 
    st->add_inst(); 

    if(st->level > 0) std::cout << "] U4Tree::Create " << std::endl ; 
    return tr ; 
}

inline U4Tree::U4Tree(stree* st_, const G4VPhysicalVolume* const top_,  const U4SensorIdentifier* sid_ )
    :
    st(st_),
    top(top_),
    sid(sid_ ? sid_ : new U4SensorIdentifierDefault),
    level(st->level),
    num_surfaces(-1),
    rayleigh_table(CreateRayleighTable())
{
    init(); 
}


inline void U4Tree::init()
{
    if(top == nullptr) return ; 

    initDomain(); 
    initRayleigh(); 
    initMaterials();
    initMaterials_NoRINDEX(); 

    initSurfaces();
    initSolids();
    initNodes(); 
    initSurfaces_Serialize();

    initBoundary(); // Currently EMPTY IMPL
}

inline void U4Tree::initDomain()
{
    sdomain dom ; 
    st->wavelength = dom.get_wavelength_nm() ; 
    st->energy = dom.get_energy_eV() ; 
}

/**
U4Tree::initMaterials
-----------------------

Canonically invoked from U4Tree::init 

1. recursive traverse collecting material pointers from all active LV into materials vector 
   in postorder of first encounter.

2. creates SSim/stree/material holding properties of all active materials::

CONSIDERING : maybe relocate to SSim/material ? rather than SSim/stree/material ? 
and hold the material NPFold member in SSim ? 


The creation of the standard *stree::mat* array using U4Material::MakeStandardArray
gets most material properties from the MPTs of the materials. However 
as G4OpRayleigh does some sneaky generation of RAYLEIGH scatter props
in its physics table some overrides are done getting Water/RAYLEIGH
from rayleigh_table.    

**/

inline void U4Tree::initMaterials()
{
    initMaterials_r(top); 
    st->material = U4Material::MakePropertyFold(materials);  


    G4PhysicsVector* prop = rayleigh_table->find("Water") ;  
    assert( prop ); 
    std::map<std::string, G4PhysicsVector*> prop_override ; 
    prop_override["Water/RAYLEIGH"] = prop ; 

    st->mat = U4Material::MakeStandardArray(materials, prop_override) ; 
}

inline void U4Tree::initMaterials_NoRINDEX()
{
    int num_materials = materials.size() ; 
    for(int i=0 ; i < num_materials ; i++)
    {
        const G4Material* mt = materials[i] ;    
        const G4MaterialPropertyVector* rindex = GetRINDEX( mt ) ; 
        if( rindex == nullptr )
        {
            const char* mtn = mt->GetName().c_str(); 
            st->mtname_no_rindex.push_back(mtn) ; 
        }
    }
}


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

Trying to find pre-existing G4OpRayleigh process
with the argumentless U4PhysicsTable ctor fails 
when U4Tree instanciation happens where it does currently.  
As a workaround try passing in a throwaway G4OpRayleigh 
just to get access to its physics table. 

**/

inline void U4Tree::initRayleigh()
{
    std::cerr 
        << "U4Tree::initRayleigh" 
        << ( rayleigh_table ? rayleigh_table->desc() : "-" ) 
        ;

    st->rayleigh = rayleigh_table ? rayleigh_table->tab : nullptr  ; 
}


inline void U4Tree::initMaterials_r(const G4VPhysicalVolume* const pv)
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ;i++ ) initMaterials_r( lv->GetDaughter(i) ); 
    G4Material* mt = lv->GetMaterial() ; // postorder visit after recursive call  
    if(mt && (std::find(materials.begin(), materials.end(), mt) == materials.end())) initMaterial(mt);  
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
    num_surfaces = int(surfaces.size()) ; 

    for(int i=0 ; i < num_surfaces ; i++)
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

As this requires to run after implicit are 
collected in initNodes it is too soon to do 
this within initSurfaces

**/

inline void U4Tree::initSurfaces_Serialize()
{
    std::vector<U4SurfacePerfect> perfect ; 
    U4SurfacePerfect::Get(perfect);

    U4SurfaceArray serialize(surfaces, st->implicit, perfect) ;   
    st->sur = serialize.sur ; 

    for(int i=0 ; i < serialize.num_perfect ; i++)
    {
        const U4SurfacePerfect& perf = perfect[i] ; 
        const char* name = perf.name.c_str() ; 
        st->add_surface( name );   
    }

    // ITS TOO LATE ADD IMPLICIT NAMES HERE 
    // AS THOSE ARE NEEDED FOR stree::get_boundary_name 
    // CAN ADD PERFECTS AS THOSE ARE JUST FOR TESTING
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

**/

inline void U4Tree::initSolid(const G4VSolid* const so, int lvid )
{
    assert( int(solids.size()) == lvid ); 

    int root = U4Solid::Convert(so, lvid, 0 ); 
    assert( root > -1 ); 
    snd::SetLVID(root, lvid ); 

    G4String _name = so->GetName() ; // bizarre: G4VSolid::GetName returns by value, not reference
    const char* name = _name.c_str();    

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

Issue : omits to do the equivalent of X4PhysicalVolume::convertImplicitSurfaces_r 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* see ~/opticks/notes/issues/stree_bd_names_and_Implicit_RINDEX_NoRINDEX.rst
* causes difference between SSim/bnd_names.txt and SSim/stree/bd_names.txt

All transparent materials like Scintillator, Acrylic, Water should have RINDEX property. 
Some absorbing materials like Tyvek might not have RINDEX property as 
lazy Physicists sometimes rely on sloppy Geant4 implicit behavior 
which causes fStopAndKill at the RINDEX->NoRINDEX boundary
as if there was a perfect absorbing surface there.  

To mimic the implicit surface Geant4 behaviour with Opticks on GPU 
it is necessary to add explicit perfect absorber surfaces. 

First try at implicit handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HMM: want to mint only the minimum number of implicits
so need to define a name that captures the desired identity 
and only collect more implicits when they have different names. 

Implicit border surface "directionality" is always 
from the material with RINDEX to the material without RINDEX 

U4SurfaceArray is assuming the implicits all appear together
after standard surfaces and before perfects. 
All standard surfaces are collected in initSurfaces so 
have a constant base of surfaces ontop of which to 
add implicits. 

When an implicit is detected the osur or isur is 
changed accordingly.   

**/


struct U4TreeBorder
{
    stree* st ; 
    int num_surfaces ; 

    const G4LogicalVolume* const lv ; 
    const G4LogicalVolume* const lv_p ;  
    const G4Material* const imat_ ; 
    const G4Material* const omat_ ; 
    const char* imatn ; 
    const char* omatn ; 
    const std::string& inam ; 
    const std::string& onam ; 
    const G4LogicalSurface* const osur_ ; 
    const G4LogicalSurface* const isur_ ;  
    const G4MaterialPropertyVector* i_rindex ; 
    const G4MaterialPropertyVector* o_rindex ; 

    int  implicit_idx ; 
    bool implicit_isur ; 
    bool implicit_osur ; 

    U4TreeBorder(
        stree* st_, 
        int num_surfaces_, 
        const G4VPhysicalVolume* const pv, 
        const G4VPhysicalVolume* const pv_p 
        ); 

    void init(); 
    void maybe_implicit_override( int4& bd ); 

}; 

inline U4TreeBorder::U4TreeBorder(
        stree* st_, 
        int num_surfaces_, 
        const G4VPhysicalVolume* const pv, 
        const G4VPhysicalVolume* const pv_p )
    :
    st(st_),
    num_surfaces(num_surfaces_),
    lv(pv->GetLogicalVolume()),
    lv_p(pv_p ? pv_p->GetLogicalVolume() : lv),
    imat_(lv->GetMaterial()),
    omat_(lv_p ? lv_p->GetMaterial() : imat_), // top omat -> imat 
    imatn(imat_->GetName().c_str()),
    omatn(omat_->GetName().c_str()),
    inam(pv->GetName()), 
    onam(pv_p ? pv_p->GetName() : inam), 
    osur_(U4Surface::Find( pv_p, pv )),    // look for border or skin surface
    isur_(U4Surface::Find( pv  , pv_p )),
    i_rindex(U4Tree::GetRINDEX( imat_ )), 
    o_rindex(U4Tree::GetRINDEX( omat_ )),
    implicit_idx(-1),
    implicit_isur(i_rindex != nullptr && o_rindex == nullptr),   // now just supects 
    implicit_osur(o_rindex != nullptr && i_rindex == nullptr)
{
    init(); 
}

/**
HMM: what about 2 implicits at once ? Its highly unlikely, very broken geom.

TODO : finesse implicit judgement based on preexisting surfaceMPT + finish  

Suspect should only set implicits when there is no 
corresponding preexisting surface. 

That would mean moving the init into the maybe ...
Also split isur from osur judgement 

**/

inline void U4TreeBorder::init()
{
    implicit_idx = -1 ; 

    if(implicit_isur || implicit_osur )  
    {
        bool flip = implicit_osur ; 
        //std::string implicit_ = S4::ImplicitBorderSurfaceName(inam, onam, flip );  
        std::string implicit_ = S4::ImplicitBorderSurfaceName(inam, imatn, onam, omatn, flip );  
        const char* implicit = implicit_.c_str(); 

        bool new_implicit = U4Tree::GetValueIndex<std::string>( st->implicit, implicit ) == -1 ;  
        if(new_implicit)
        {
            st->implicit.push_back(implicit) ;   
            st->add_surface(implicit);   
        }
        implicit_idx = U4Tree::GetValueIndex<std::string>( st->implicit, implicit ) ; 
    }
}

inline void U4TreeBorder::maybe_implicit_override( int4& bd )
{
    int& osur = bd.y ; 
    int& isur = bd.z ; 

    if( implicit_idx > -1 )
    {
        if(implicit_isur) // from imat to omat : outwards
        { 
            if( isur != -1 ) std::cerr 
                << "U4TreeBorder::maybe_implicit_override"
                << " changing isur from " << isur 
                << " to " << ( num_surfaces + implicit_idx )
                << " num_surfaces " << num_surfaces 
                << " border.implicit_idx " << implicit_idx
                << std::endl 
                ;

            st->implicit_isur.push_back(bd);  
            isur = num_surfaces + implicit_idx ; 
            st->implicit_isur.push_back(bd);  
        }
        else if(implicit_osur) // from omat to imat : inwards
        {
            //assert(osur == -1 );           // loads of these
            if( osur != -1 ) std::cerr 
                << "U4TreeBorder::maybe_implicit_override"
                << " changing osur from " << osur 
                << " to " << ( num_surfaces + implicit_idx )
                << " num_surfaces " << num_surfaces 
                << " implicit_idx " << implicit_idx
                << std::endl 
                ;

            st->implicit_osur.push_back(bd);  
            osur = num_surfaces + implicit_idx ; 
            st->implicit_osur.push_back(bd);  
        }
    }
}




inline int U4Tree::initNodes_r( const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p, int depth, int sibdex, int parent )
{

    U4TreeBorder border(st, num_surfaces, pv, pv_p) ; 

    const G4LogicalVolume* const lv = pv->GetLogicalVolume();
/*
    const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : nullptr ;

    const G4Material* const imat_ = lv->GetMaterial() ;
    const G4Material* const omat_ = lv_p ? lv_p->GetMaterial() : imat_ ;  // top omat -> imat 

    const char* imatn = imat_->GetName().c_str() ; 
    const char* omatn = omat_->GetName().c_str() ; 

    const std::string& inam = pv->GetName() ; 
    const std::string& onam = pv_p ? pv_p->GetName() : pv->GetName() ;  

    const G4LogicalSurface* const osur_ = U4Surface::Find( pv_p, pv ); 
    const G4LogicalSurface* const isur_ = U4Surface::Find( pv  , pv_p ); 

    const G4MaterialPropertyVector* i_rindex = GetRINDEX( imat_ ) ; 
    const G4MaterialPropertyVector* o_rindex = GetRINDEX( omat_ ) ; 
*/


   
    int imat = GetPointerIndex<G4Material>(materials, border.imat_); 
    int omat = GetPointerIndex<G4Material>(materials, border.omat_); 
    int isur = GetPointerIndex<G4LogicalSurface>(surfaces, border.isur_); 
    int osur = GetPointerIndex<G4LogicalSurface>(surfaces, border.osur_); 

    int4 bd = {omat, osur, isur, imat } ; 
    border.maybe_implicit_override(bd) ; 

    bool new_boundary = GetValueIndex<int4>( st->bd, bd ) == -1 ; 
    if(new_boundary)  
    {
        st->bd.push_back(bd) ; 
        std::string bdn = st->get_boundary_name(bd,'/') ; 
        st->bdname.push_back(bdn.c_str()) ; 
        // HMM: better to use higher level stree::add_boundary if can get names at stree level 
    }
    int boundary = GetValueIndex<int4>( st->bd, bd ) ; 
    assert( boundary > -1 ); 


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

    nd.repeat_ordinal = -1 ;  // changed later for instance subtrees by stree::labelFactorSubtrees
    nd.boundary = boundary ; 


    pvs.push_back(pv); 

    st->nds.push_back(nd); 
    st->digs.push_back(dig); 
    st->m2w.push_back(tr_m2w);  
    st->w2m.push_back(tr_w2m);  



    glm::tmat4x4<double> tt_gtd(1.) ;    // "GGeo Transform Debug" comparison
    glm::tmat4x4<double> vv_gtd(1.) ;

    bool local = false ; 
    bool reverse = false ; 
    st->get_node_product( tt_gtd, vv_gtd, nidx, local, reverse, nullptr );   
    // product of m2w transforms from root down to nidx,  must be after push_backs of nd and tr_m2w

    st->gtd.push_back(tt_gtd);  


    if(sibdex == 0 && nd.parent > -1) st->nds[nd.parent].first_child = nd.index ; 
    // record first_child nidx into parent snode by reaching up thru the recursion levels 

    int p_sib = -1 ; 
    int i_sib = -1 ; 
    for (int i=0 ; i < num_child ;i++ ) 
    {
        p_sib = i_sib ;    // node index of previous child gets set for i > 0

        //                    ch_pv ch_parent_pv ch_depth ch_sibdex ch_parent    
        i_sib = initNodes_r( lv->GetDaughter(i), pv, depth+1, i, nd.index ); 

        if(p_sib > -1) st->nds[p_sib].next_sibling = i_sib ; 
        // after first child : reach back to previous sibling snode to set the sib->sib linkage, default -1
    }

    return nd.index ; 
}






/**
U4Tree::initBoundary
-----------------------

Note that most of the boundary preparation happens in initNodes
as want to store boundary int with the snode. 

TODO: what about the implicit surfaces


Currently using an unholy mixture of old and new:

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

inline void U4Tree::initBoundary()
{


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

Initially tried to simply use lv->GetSensitiveDetector() to 
identify sensor nodes by that is problematic because 
the SD is not on the volume with the copyNo and this 
use of copyNo is detector specific.  Also not all JUNO SD
are actually sensitive. 

**/

inline void U4Tree::identifySensitive()
{
    if(level > 0) std::cerr << "[ U4Tree::identifySensitive " << std::endl ; 
    st->sensor_count = 0 ; 

    identifySensitiveInstances(); 
    identifySensitiveGlobals(); 
    st->reorderSensors();  // change nd.sensor_index to facilitate comparison with GGeo

    if(level > 0) std::cerr << "] U4Tree::identifySensitive st.sensor_count " << st->sensor_count << std::endl ; 
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
    if(level > 0) std::cerr 
        << "[ U4Tree::identifySensitiveInstances" 
        << " num_factor " << num_factor 
        << " st.sensor_count " << st->sensor_count 
        << std::endl
        ; 

    for(unsigned i=0 ; i < num_factor ; i++)
    {
        std::vector<int> outer ; 
        st->get_factor_nodes(outer, i );  // nidx of outer volumes of instances 
        sfactor& fac = st->get_factor_(i); 
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

        if(level > 1) std::cerr
            << "U4Tree::identifySensitiveGlobals"
            << " i " << std::setw(7) << i 
            << " sensor_id " << std::setw(7) << sensor_id  
            << " sensor_index " << std::setw(7) << sensor_index  
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
        G4String name = solid->GetName(); 
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


