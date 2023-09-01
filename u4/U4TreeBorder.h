#pragma once
/**
U4TreeBorder.h : Implicit surface handling : TODO : pick a better name 
=========================================================================

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

Confusing and pointless osur from absorbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

see ~/opticks/notes/issues/optical_ems_4_getting_too_many_from_non_sensor_Vacuum_Steel_borders.rst


**/

#include "ssys.h"
#include "U4Mat.h"


struct U4TreeBorder 
{
    stree* st ; 

    const G4LogicalVolume* const lv ; 
    const G4LogicalVolume* const lv_p ;  
    const G4Material* const imat_ ; 
    const G4Material* const omat_ ; 
    const char* imat ; 
    const char* omat ; 
    const G4VSolid* isolid_ ; 
    const G4VSolid* osolid_ ;
    G4String isolid ; // unavoidable, because G4VSolid::GetName returns by value 
    G4String osolid ; 
    const char* _isolid ;  
    const char* _osolid ; 
    const std::string& inam ; 
    const std::string& onam ; 

    const G4MaterialPropertyVector* i_rindex ; 
    const G4MaterialPropertyVector* o_rindex ; 

    const G4LogicalSurface* const osur_ ; 
    const G4LogicalSurface* const isur_ ;  

    int  implicit_idx ; 
    bool implicit_isur ; 
    bool implicit_osur ; 

    const char* flagged_isolid ; 

    U4TreeBorder(
        stree* st_, 
        const G4VPhysicalVolume* const pv, 
        const G4VPhysicalVolume* const pv_p 
        ); 

    std::string desc() const ; 
    bool is_flagged() const ; 

    int  get_surface_idx_of_implicit(bool flip); 

    bool has_osur_override( const int4& bd ) const ; 
    bool has_isur_override( const int4& bd ) const ; 
    void do_osur_override( int4& bd ) ; 
    void do_isur_override( int4& bd ) ; 

    void check( const int4& bd )  ; 
}; 

/**
U4TreeBorder::U4TreeBorder
-----------------------------

Boundary spec::

   omat/osur/isur/imat 
    
   +-----------------------------+
   |                             |
   | pv_p    omat    osolid      |
   |              ||             |     implicit_osur : o_rindex != nullptr && i_rindex = nullptr
   |              ||             | 
   |         osur \/             |     osur : relevant for photons going from pv_p -> pv ( outer to inner, ingoing )
   +-----------------------------+ 
   |         isur /\             |     isur : relevant for photons goes from pv -> pv_p  ( inner to outer, outgoing ) 
   |              ||             | 
   |              ||             |     implicit_isur : i_rindex != nullptr && o_rindex == nullptr
   | pv      imat    isolid      |
   |                             |
   +-----------------------------+

**/


inline U4TreeBorder::U4TreeBorder(
    stree* st_, 
    const G4VPhysicalVolume* const pv, 
    const G4VPhysicalVolume* const pv_p )
    :
    st(st_),
    lv(pv->GetLogicalVolume()),
    lv_p(pv_p ? pv_p->GetLogicalVolume() : lv),
    imat_(lv->GetMaterial()),
    omat_(lv_p ? lv_p->GetMaterial() : imat_), // top omat -> imat 
    imat(imat_->GetName().c_str()),
    omat(omat_->GetName().c_str()),
    isolid_(lv->GetSolid()),
    osolid_(lv_p->GetSolid()),
    isolid(isolid_->GetName()),
    osolid(osolid_->GetName()),
    _isolid(isolid.c_str()),
    _osolid(osolid.c_str()),
    inam(pv->GetName()), 
    onam(pv_p ? pv_p->GetName() : inam), 
    i_rindex(U4Mat::GetRINDEX( imat_ )), 
    o_rindex(U4Mat::GetRINDEX( omat_ )),
    osur_(U4Surface::Find( pv_p, pv )),    // look for border or skin surface, HMM: maybe disable for o_rindex == nullptr ?
    isur_( i_rindex == nullptr ? nullptr : U4Surface::Find( pv, pv_p )), // disable isur from absorbers without RINDEX
    implicit_idx(-1),
    implicit_isur(i_rindex != nullptr && o_rindex == nullptr),  
    implicit_osur(o_rindex != nullptr && i_rindex == nullptr),
    flagged_isolid(ssys::getenvvar("U4TreeBorder__FLAGGED_ISOLID", "sStrutBallhead"))
{
}


std::string U4TreeBorder::desc() const 
{
    std::stringstream ss ; 
    ss << "U4TreeBorder::desc" << std::endl 
       << " omat " << omat << std::endl 
       << " imat " << imat << std::endl 
       << " osolid " << osolid << std::endl 
       << " isolid " << isolid << std::endl 
       << " is_flagged " << ( is_flagged() ? "YES" : "NO " ) 
       ;
    std::string str = ss.str(); 
    return str ; 
}

bool U4TreeBorder::is_flagged() const 
{
    return _isolid && flagged_isolid && strcmp(_isolid, flagged_isolid ) == 0 ;  
}




/**
U4TreeBorder::get_surface_idx_of_implicit
-------------------------------------------

+---------------------------------+-----------------+
|  Callers                        |  flip           |
+=================================+=================+ 
|  U4TreeBorder::do_osur_override |  true           |
+---------------------------------+-----------------+  
|  U4TreeBorder::do_isur_override |  false          |
+---------------------------------+-----------------+  


1. uses inam and onam pv name to form a name for the implicit surface
2. adds implicit name to the stree suname vector if not already present 
   and returns the standard surface index  


**/

inline int U4TreeBorder::get_surface_idx_of_implicit(bool flip)
{
    std::string implicit_ = S4::ImplicitBorderSurfaceName(inam, onam, flip );   // based on pv pv_p names
    const char* implicit = implicit_.c_str(); 
    int surface_idx = st->add_surface_implicit(implicit) ;  // now returns standard surface idx 
    return surface_idx ; 
}

/**
U4TreeBorder::has_osur_override
--------------------------------

Only returns true when:

1. materials are RINDEX->NoRINDEX 
2. AND no corresponding surface defined already 

The old X4/GGeo workflow does similar to 
this in X4PhysicalVolume::convertImplicitSurfaces_r
but in addition that workflow skips osur, doing only isur
for no valid reason that I can find/recall. 

**/

inline bool U4TreeBorder::has_osur_override( const int4& bd ) const 
{
    const int& osur = bd.y ; 
    return osur == -1 && implicit_osur == true ;  
}
inline bool U4TreeBorder::has_isur_override( const int4& bd ) const 
{
    const int& isur = bd.z ; 
    return isur == -1 && implicit_isur == true ;  
}

/**
U4TreeBorder::do_osur_override U4TreeBorder::do_isur_override
--------------------------------------------------------------

Called from U4Tree::initNodes_r when osur/isur implicits are enabled
and the above has override methods return true. 

**/

inline void U4TreeBorder::do_osur_override( int4& bd ) // from omat to imat : inwards
{
    int& osur = bd.y ; 
    osur = get_surface_idx_of_implicit(true); 
}
inline void U4TreeBorder::do_isur_override( int4& bd ) // from imat to omat : outwards
{
    int& isur = bd.z ; 
    isur = get_surface_idx_of_implicit(false); 
}

inline void U4TreeBorder::check( const int4& bd ) 
{
    if(is_flagged()) std::cout 
        << "U4TreeBorder::check is_flagged " << std::endl 
        << " (omat,osur,isur,imat) " << bd << std::endl 
        << desc()
        << std::endl 
        ;
}


