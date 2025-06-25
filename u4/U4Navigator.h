#pragma once
/**
U4Navigator.h
===============


**/

#include <cassert>
#include <iostream>

#include "scuda.h"
#include "squad.h"

#include "G4Navigator.hh"
#include "G4TransportationManager.hh"
#include "U4Tree.h"


struct U4Navigator_Stats
{
    int num_call = 0 ;
    int num_nullpv0 = 0 ;
    int num_withpv0 = 0 ;

    int num_isect = 0 ;
    int num_nullpv1 = 0 ;
    int num_withpv1 = 0 ;

    std::string desc() const ;
};

inline std::string U4Navigator_Stats::desc() const
{
    double frac_nullpv0 = double(num_nullpv0)/double(num_call) ;
    double frac_nullpv1 = double(num_nullpv1)/double(num_isect) ;
    std::stringstream ss ;
    ss
        << " num_call " << num_call
        << " num_isect " << num_isect
        << " num_nullpv0 " << num_nullpv0
        << " num_withpv0 " << num_withpv0
        << " num_nullpv1 " << num_nullpv1
        << " num_withpv1 " << num_withpv1
        << " frac_nullpv0 " << std::setw(10) << std::setprecision(3) << std::fixed << frac_nullpv0
        << " frac_nullpv1 " << std::setw(10) << std::setprecision(3) << std::fixed << frac_nullpv1
        ;

    std::string str = ss.str() ;
    return str ;
}


struct U4Navigator_Intersect
{
    G4ThreeVector ipos = {} ;
    G4VPhysicalVolume* pv0 = nullptr ;
    G4VPhysicalVolume* pv1 = nullptr ;
    G4LogicalVolume* lv1 = nullptr ;
    G4VSolid* so1 = nullptr ;
    G4double t = 0. ;
    G4bool valid = false ;
    int nidx = -10 ;
    int prim = -10 ;
    const char* prn = nullptr ;

    void zero();
    std::string desc() const ;
};


inline void U4Navigator_Intersect::zero()
{
    ipos = {} ;
    pv0 = nullptr ;
    pv1 = nullptr ;
    lv1 = nullptr ;
    so1 = nullptr ;
    t = 0 ;
    valid = false ;
    nidx = -10 ;
    prim = -10 ;
    prn = nullptr ;
}

inline std::string U4Navigator_Intersect::desc() const
{
    std::stringstream ss ;
    ss
        << " t " << std::setw(10) << std::fixed << std::setprecision(3) << t
        << " valid "  << ( valid ? "YES" : "NO " )
        << " nidx "  << std::setw(8) << nidx
        << " prim "  << std::setw(6) << prim
        << " prn " << ( prn ? prn : "-" )
        << " pv0 " << ( pv0 ? pv0->GetName() : "-" )
        << " pv1 " << ( pv1 ? pv1->GetName() : "-" )
        << " so1 " << ( so1 ? so1->GetName() : "-" )
        ;

    std::string str = ss.str() ;
    return str ;
}



struct U4Navigator
{
    static G4Navigator* GetNav();
    static double Distance(const G4ThreeVector& ori, const G4ThreeVector& dir );


    U4Navigator(const U4Tree* tree);

    const U4Tree* tree ;
    G4Navigator* nav ;
    U4Navigator_Stats stats = {} ;
    U4Navigator_Intersect isect = {} ;

    void simtrace(quad4& p );
    void getIntersect( const G4ThreeVector& ori, const G4ThreeVector& dir );
};

inline G4Navigator* U4Navigator::GetNav()
{
    G4Navigator* _nav = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    return _nav ;
}


inline double U4Navigator::Distance( const G4ThreeVector& ori, const G4ThreeVector& dir )
{
    U4Navigator nav(nullptr);
    nav.getIntersect( ori, dir);
    return nav.isect.t ;
}



inline U4Navigator::U4Navigator(const U4Tree* _tree)
    :
    tree(_tree),
    nav(nullptr)  // GetNav when needed
{
}


/**
U4Navigator::getIntersect
---------------------------

Canonical caller U4Navigator::simtrace

**/


inline void U4Navigator::getIntersect( const G4ThreeVector& ori, const G4ThreeVector& dir )
{
    isect.zero();

    if( nav == nullptr ) nav = GetNav();

    G4VPhysicalVolume* pv0 = nullptr ;
    {
        const G4bool pRelativeSearch=false ;
        const G4bool ignoreDirection=false ;
        pv0 = nav->LocateGlobalPointAndSetup(ori, &dir, pRelativeSearch, ignoreDirection );
    }
    stats.num_call++ ;
    if(pv0 == nullptr) stats.num_nullpv0++ ;
    if(pv0 != nullptr) stats.num_withpv0++ ;


    G4double t ;
    {
        const G4double pCurrentProposedStepLength = kInfinity ;
        G4double pNewSafety ;
        t = nav->ComputeStep(ori, dir, pCurrentProposedStepLength, pNewSafety);
    }

    G4bool valid = t > 0. && t != kInfinity ;

    isect.pv0 = pv0 ;
    isect.t = t ;
    isect.valid = valid ;

    assert(tree);

    if( valid )
    {
        G4ThreeVector ipos = ori + t*dir ;
        G4VPhysicalVolume* pv1 = nav->LocateGlobalPointAndSetup(ipos, nullptr, false, true );
        G4LogicalVolume* lv1 = pv1 ? pv1->GetLogicalVolume() : nullptr ;
        G4VSolid* so1 = lv1 ? lv1->GetSolid() : nullptr ;

        isect.ipos = ipos ;
        isect.pv1 = pv1 ;
        isect.lv1 = lv1 ;
        isect.so1 = so1 ;


        // try to identify the intersected volume
        int nidx        = pv1  == nullptr ? -1      : tree->get_nidx( pv1 )           ; // look for pv1 in stree::pvs vector
        int prim        = nidx == -1      ? -1      : tree->get_prim_for_nidx( nidx ) ;
        const char* prn = prim == -1      ? nullptr : tree->get_prname( prim )        ;

        isect.nidx = nidx ;
        isect.prim = prim ;
        isect.prn = prn ;

        stats.num_isect += 1;
        if(isect.pv1 == nullptr) stats.num_nullpv1++ ;
        if(isect.pv1 != nullptr) stats.num_withpv1++ ;
    }

}

/**
U4Navigator::simtrace
-----------------------

Canonical callstack::

    U4Simtrace::EndOfRunAction > U4Simtrace::Scan

Assuming standard simtrace layout (see sevent::add_simtrace and below)
this uses U4Navigator::Distance to populate the intersect position.
HMM: how to get the surface normal at the intersect position ?

* will need to get the G4VSolid ?

* q0.f.xyz normal at intersect (not implemented) q0.f.z distance to intersect
* q1.f.xyz intersect position    q1.f.w tmin
* q2.f.xyz trace origin
* q3.f.xyz trace direction

The simtrace layout is documented in sevent::add_simtrace

**/

inline void U4Navigator::simtrace( quad4& p )
{
    G4ThreeVector ori(p.q2.f.x, p.q2.f.y, p.q2.f.z);
    G4ThreeVector dir(p.q3.f.x, p.q3.f.y, p.q3.f.z);

    getIntersect(ori, dir);

    p.q0.f.x = 0.f ; // not yet implemented : surface normal at intersect
    p.q0.f.y = 0.f ;
    p.q0.f.z = 0.f ;
    p.q0.f.w = isect.t ;

    p.q1.f.x = float(isect.ipos.x()) ;
    p.q1.f.y = float(isect.ipos.y()) ;
    p.q1.f.z = float(isect.ipos.z()) ;
    p.q1.f.w = 0.f  ; // tmin

    unsigned globalPrimIdx = isect.prim == -1 ? 0xffffu : ( isect.prim & 0xffffu ) ;
    // Formerly isect.nidx but as that is not equivalent globalPrimIdx
    // so have added U4Tree/stree functionality to get the globalPrimIdx
    // rather than CSGFoundry which is the natural place to get globalPrimIdx from.

    unsigned boundary = 0u & 0xffffu ;
    unsigned globalPrimIdx_boundary = ( globalPrimIdx << 16 ) | boundary  ;
    unsigned identity = 0u ;

    p.q2.u.w = globalPrimIdx_boundary ;
    p.q3.u.w = identity ;
}


