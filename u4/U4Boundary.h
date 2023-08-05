#pragma once

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"

#include "sstr.h"
#include "U4Mat.h"
#include "U4Surface.h"

struct U4Boundary
{
    const G4LogicalVolume* const lv ; 
    const G4LogicalVolume* const lv_p ;  
    const G4Material* const imat_ ; 
    const G4Material* const omat_ ; 
    const G4MaterialPropertyVector* i_rindex ; 
    const G4MaterialPropertyVector* o_rindex ; 
    const char* imat ; 
    const char* omat ; 
    const G4LogicalSurface* const osur_ ; 
    const G4LogicalSurface* const isur_ ;  
    const char* isur ; 
    const char* osur ; 
    std::string bnd  ; 
 
    U4Boundary(
        const G4VPhysicalVolume* const pv, 
        const G4VPhysicalVolume* const pv_p 
        ); 

};

inline U4Boundary::U4Boundary(
        const G4VPhysicalVolume* const pv, 
        const G4VPhysicalVolume* const pv_p 
        )
    :
    lv(pv->GetLogicalVolume()),
    lv_p(pv_p ? pv_p->GetLogicalVolume() : lv),
    imat_(lv->GetMaterial()),
    omat_(lv_p ? lv_p->GetMaterial() : imat_), // top omat -> imat 
    i_rindex(U4Mat::GetRINDEX( imat_ )), 
    o_rindex(U4Mat::GetRINDEX( omat_ )),
    imat(imat_->GetName().c_str()),
    omat(omat_->GetName().c_str()),
    osur_(U4Surface::Find( pv_p, pv )),    // look for border or skin surface
    isur_( i_rindex == nullptr ? nullptr : U4Surface::Find( pv  , pv_p )), // disable isur from absorbers without RINDEX
    isur(isur_ ? isur_->GetName().c_str() : nullptr),
    osur(osur_ ? osur_->GetName().c_str() : nullptr),
    bnd(sstr::Join("/",omat,osur,isur,imat))
{
}


inline std::ostream& operator<<(std::ostream& os, const U4Boundary& b)  
{
    os << "U4Boundary " << b.bnd ; 
    return os; 
}


