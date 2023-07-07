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

**/

#include "U4Mat.h"


struct U4TreeBorder 
{
    stree* st ; 
    int num_surfaces ; 

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

    std::string desc() const ; 
    bool is_flagged() const ; 

    int  get_override_idx(bool flip); 
    bool has_osur_override( const int4& bd ) const ; 
    bool has_isur_override( const int4& bd ) const ; 
    void do_osur_override( int4& bd ) ; 
    void do_isur_override( int4& bd ) ; 


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
    osur_(U4Surface::Find( pv_p, pv )),    // look for border or skin surface
    isur_(U4Surface::Find( pv  , pv_p )),
    i_rindex(U4Mat::GetRINDEX( imat_ )), 
    o_rindex(U4Mat::GetRINDEX( omat_ )),
    implicit_idx(-1),
    implicit_isur(i_rindex != nullptr && o_rindex == nullptr),   // now just supects 
    implicit_osur(o_rindex != nullptr && i_rindex == nullptr)
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
    return _isolid && strcmp(_isolid, "sStrutBallhead") == 0 ;  
}




/**
U4TreeBorder::get_override_idx
-------------------------------

When debugging use alternate verbose implicit name.

**/

inline int U4TreeBorder::get_override_idx(bool flip)
{
    std::string implicit_ = S4::ImplicitBorderSurfaceName(inam, onam, flip );  
    //std::string implicit_ = S4::ImplicitBorderSurfaceName(inam, imatn, onam, omatn, flip );  

    const char* implicit = implicit_.c_str(); 

    int implicit_idx = stree::GetValueIndex<std::string>( st->implicit, implicit ) ; 
    if(implicit_idx == -1)  // new implicit 
    {
        st->implicit.push_back(implicit) ;   
        st->add_surface(implicit);   
        implicit_idx = stree::GetValueIndex<std::string>( st->implicit, implicit ) ; 
    }
    assert( implicit_idx > -1 ); 
    return num_surfaces + implicit_idx ;
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
    //return implicit_osur == true ;    // old logic, giving too many overrides 
}
inline bool U4TreeBorder::has_isur_override( const int4& bd ) const 
{
    const int& isur = bd.z ; 
    return isur == -1 && implicit_isur == true ;  
    //return implicit_isur == true ;   // old logic, giving too many overrides
}
inline void U4TreeBorder::do_osur_override( int4& bd ) // from omat to imat : inwards
{
#ifdef DEBUG_IMPLICIT
    st->implicit_osur.push_back(bd);  
#endif
    int& osur = bd.y ; 
    osur = get_override_idx(true); 
#ifdef DEBUG_IMPLICIT
    st->implicit_osur.push_back(bd);  
#endif
}
inline void U4TreeBorder::do_isur_override( int4& bd ) // from imat to omat : outwards
{
#ifdef DEBUG_IMPLICIT
    st->implicit_isur.push_back(bd); 
#endif
    int& isur = bd.z ; 
    isur = get_override_idx(false); 
#ifdef DEBUG_IMPLICIT
    st->implicit_isur.push_back(bd); 
#endif
}






