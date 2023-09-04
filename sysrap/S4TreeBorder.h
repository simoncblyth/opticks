#pragma once
/**
S4TreeBorder.h
===============

Extract some of the functionality of U4TreeBorder.h whilst trying 
to get implicit osur surfaces to work in old workflow. 

**/

struct S4TreeBorder
{
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
    const std::string& inam ; // inner pv name
    const std::string& onam ; // outer pv name

    const G4MaterialPropertyVector* i_rindex ;
    const G4MaterialPropertyVector* o_rindex ;

    const G4LogicalSurface* const osur_ ;
    const G4LogicalSurface* const isur_ ;

    int  implicit_idx ;
    bool implicit_isur ;
    bool implicit_osur ;

    const char* flagged_isolid ;

    S4TreeBorder(
        const G4VPhysicalVolume* const pv,
        const G4VPhysicalVolume* const pv_p
        );

    std::string desc() const ;
    bool is_flagged() const ; 

};

inline S4TreeBorder::S4TreeBorder(
    const G4VPhysicalVolume* const pv, 
    const G4VPhysicalVolume* const pv_p )
    :   
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
    i_rindex(S4Mat::GetRINDEX( imat_ )), 
    o_rindex(S4Mat::GetRINDEX( omat_ )), 
    osur_(S4Surface::Find( pv_p, pv )),    // look for border or skin surface, HMM: maybe disable for o_rindex == nullptr ?
    isur_( i_rindex == nullptr ? nullptr : S4Surface::Find( pv, pv_p )), // disable isur from absorbers without RINDEX
    implicit_idx(-1),
    implicit_isur(i_rindex != nullptr && o_rindex == nullptr),  
    implicit_osur(o_rindex != nullptr && i_rindex == nullptr),
    flagged_isolid(ssys::getenvvar("S4TreeBorder__FLAGGED_ISOLID", "sStrutBallhead"))
{
}


inline std::string S4TreeBorder::desc() const
{
    std::stringstream ss ;
    ss << "S4TreeBorder::desc" << std::endl
       << " omat " << omat << std::endl
       << " imat " << imat << std::endl
       << " osolid " << osolid << std::endl
       << " isolid " << isolid << std::endl
       << " is_flagged " << ( is_flagged() ? "YES" : "NO " )
       ;
    std::string str = ss.str();
    return str ;
}

inline bool S4TreeBorder::is_flagged() const
{
    return _isolid && flagged_isolid && strcmp(_isolid, flagged_isolid ) == 0 ;
}



