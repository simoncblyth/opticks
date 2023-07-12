#pragma once

#include <vector>
#include <iostream>

#include "U4SensorIdentifier.h"
#include "G4PVPlacement.hh"

struct U4SensorIdentifierDefault : public U4SensorIdentifier 
{
    int getGlobalIdentity(const G4VPhysicalVolume* pv ) const ; 
    int getInstanceIdentity(const G4VPhysicalVolume* instance_outer_pv ) const ; 
    static void FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth );  
}; 


inline int U4SensorIdentifierDefault::getGlobalIdentity( const G4VPhysicalVolume* ) const 
{
    return -1 ;  
}

/**
U4SensorIdentifierDefault::getInstanceIdentity
---------------------------------------------------

Canonically used from U4Tree::identifySensitiveInstances

The argument *instance_outer_pv* is recursively traversed


**/

inline int U4SensorIdentifierDefault::getInstanceIdentity( const G4VPhysicalVolume* instance_outer_pv ) const 
{
    const char* pvn = instance_outer_pv ? instance_outer_pv->GetName().c_str() : "-" ; 
    const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(instance_outer_pv) ;
    int copyno = pvp ? pvp->GetCopyNo() : -1 ;

    std::vector<const G4VPhysicalVolume*> sdpv ; 
    FindSD_r(sdpv, instance_outer_pv, 0 );  

    unsigned num_sd = sdpv.size() ; 
    int sensor_id = num_sd == 0 ? -1 : copyno ; 

    //bool dump = copyno < 10 ; 
    //bool dump = false ; 
    //bool dump = true ;
    bool dump = num_sd > 0 ; 

 
    if(dump) std::cout 
        << "U4SensorIdentifierDefault::getIdentity" 
        << " copyno " << copyno
        << " num_sd " << num_sd
        << " sensor_id " << sensor_id 
        << " pvn " << ( pvn ? pvn : "-" )
        << std::endl 
        ;      


    return sensor_id ; 
}

/**
U4SensorIdentifierDefault::FindSD_r
-------------------------------------

Recursive traverse collecting pv pointers for pv with associated SensitiveDetector. 

**/

inline void U4SensorIdentifierDefault::FindSD_r( 
    std::vector<const G4VPhysicalVolume*>& sdpv , 
    const G4VPhysicalVolume* pv, 
    int depth )
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;
    if(sd) sdpv.push_back(pv); 
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) FindSD_r( sdpv, lv->GetDaughter(i), depth+1 );
}


