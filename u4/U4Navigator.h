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

struct U4Navigator
{
    static G4Navigator* Get();
    static G4double Distance(const G4ThreeVector &pGlobalPoint, const G4ThreeVector &pDirection, bool dump);
    static G4double Check();
    static void Simtrace( quad4& p, bool dump );
};

inline G4Navigator* U4Navigator::Get()
{
    G4Navigator* nav = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    return nav ;
}

/**
U4Navigator::Distance
----------------------

Canonical caller U4Navigator::Simtrace

**/


inline G4double U4Navigator::Distance( const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump )
{
    G4Navigator* nav = Get();

    static int num_call = 0 ;
    static int num_nullpv = 0 ;
    static int num_withpv = 0 ;

    const G4bool pRelativeSearch=false ;
    const G4bool ignoreDirection=false ;
    G4VPhysicalVolume* pv = nav->LocateGlobalPointAndSetup(pos, &dir, pRelativeSearch, ignoreDirection );

    num_call++ ;
    if(pv == nullptr) num_nullpv++ ;
    if(pv != nullptr) num_withpv++ ;

    double frac_nullpv = double(num_nullpv)/double(num_call) ;


    const G4double pCurrentProposedStepLength = kInfinity ;
    G4double pNewSafety ;
    G4double t = nav->ComputeStep(pos, dir, pCurrentProposedStepLength, pNewSafety);

    if(dump || ( pv == nullptr && num_nullpv < 10) || ( pv != nullptr && num_withpv < 10) || ( num_call % 10000 == 0) ) std::cout
        << "U4Navigator::Distance"
        << " num_call " << num_call
        << " num_nullpv " << num_nullpv
        << " num_withpv " << num_withpv
        << " frac_nullpv " << std::setw(10) << std::setprecision(3) << std::fixed << frac_nullpv
        << " pv " << ( pv ? pv->GetName() : "-" )
        << " pos " << pos
        << " dir " << dir
        << " t " << t
        << std::endl
        ;

    return t ;
}

inline G4double U4Navigator::Check()
{
    G4ThreeVector pos(0.,0.,10.);
    G4ThreeVector dir(0.,0.,1.);
    G4double t = Distance( pos, dir, true );
    return t ;
}

/**
U4Navigator::Simtrace
-----------------------

Canonical callstack::

    U4Simtrace::EndOfRunAction > U4Simtrace::Scan

Assuming standard simtrace layout (see sevent::add_simtrace and below)
this uses U4Navigator::Distance to populate the intersect position.
HMM: how to get the surface normal at the intersect position ?

* q0.f.xyz normal at intersect (not implemented) q0.f.z distance to intersect
* q1.f.xyz intersect position    q1.f.w tmin
* q2.f.xyz trace origin
* q3.f.xyz trace direction

**/

inline void U4Navigator::Simtrace( quad4& p, bool dump )
{
    G4ThreeVector ori(p.q2.f.x, p.q2.f.y, p.q2.f.z);
    G4ThreeVector dir(p.q3.f.x, p.q3.f.y, p.q3.f.z);

    G4double t = Distance(ori, dir, dump );

    if( t == kInfinity ) return ;

    G4ThreeVector ipos = ori + dir*t ;
    float tmin = 0.f ;

    p.q0.f.w = t ;

    p.q1.f.x = float(ipos.x()) ;
    p.q1.f.y = float(ipos.y()) ;
    p.q1.f.z = float(ipos.z()) ;
    p.q1.f.w = tmin  ;
}


