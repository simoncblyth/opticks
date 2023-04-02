#pragma once

#include <cassert>
#include <iostream>

#include "G4Navigator.hh"
#include "G4TransportationManager.hh"

struct U4Navigator
{
    static G4Navigator* Get(); 
    static G4double Distance(const G4ThreeVector &pGlobalPoint, const G4ThreeVector &pDirection ); 
    static G4double Check(); 
 
}; 

inline G4Navigator* U4Navigator::Get()
{
    G4Navigator* nav = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    return nav ; 
}

inline G4double U4Navigator::Distance( const G4ThreeVector &pGlobalPoint, const G4ThreeVector &pDirection )
{
    G4Navigator* nav = Get(); 

    const G4ThreeVector* direction=nullptr ; 
    const G4bool pRelativeSearch=false ; 
    const G4bool ignoreDirection=true ; 

    G4VPhysicalVolume* pv = nav->LocateGlobalPointAndSetup(pGlobalPoint, direction, pRelativeSearch, ignoreDirection );  
    assert(pv); 

    std::cout 
        << "U4Navigator::Distance"
        << " pv " << ( pv ? pv->GetName() : "-" )
        << std::endl
        ;

    const G4double pCurrentProposedStepLength = kInfinity ;     
    G4double pNewSafety ; 
    G4double dist = nav->ComputeStep(pGlobalPoint, pDirection, pCurrentProposedStepLength, pNewSafety);

    return dist ; 
}

inline G4double U4Navigator::Check()
{
    G4ThreeVector pos(0.,0.,10.); 
    G4ThreeVector dir(0.,0.,1.); 
    G4double dist = U4Navigator::Distance( pos, dir ); 
    std::cout 
        << "U4Navigator::Check"
        << " pos " << pos
        << " dir " << dir
        << " dist " << dist 
        << std::endl
        ; 

    return dist ; 
}


