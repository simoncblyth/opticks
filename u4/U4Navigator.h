#pragma once

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

inline G4double U4Navigator::Distance( const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump )
{
    G4Navigator* nav = Get();

    static int call = 0 ; 
    call++ ;  

    static int nullpv = 0 ; 

    const G4bool pRelativeSearch=false ; 
    const G4bool ignoreDirection=false ; 

    G4VPhysicalVolume* pv = nav->LocateGlobalPointAndSetup(pos, &dir, pRelativeSearch, ignoreDirection );  
    if(pv == nullptr) nullpv++ ; 
    const G4double pCurrentProposedStepLength = kInfinity ;     

    G4double pNewSafety ; 
    G4double t = nav->ComputeStep(pos, dir, pCurrentProposedStepLength, pNewSafety);

    if(dump || pv == nullptr) std::cout 
        << "U4Navigator::Distance"
        << " call " << call 
        << " nullpv " << nullpv 
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

Standard simtrace layout:

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

    // TODO: surface normal at intersect position
    p.q0.f.w = t ;

    p.q1.f.x = float(ipos.x()) ;
    p.q1.f.y = float(ipos.y()) ;
    p.q1.f.z = float(ipos.z()) ;
    p.q1.f.w = tmin  ;
}


