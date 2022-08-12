#pragma once

#include <vector>
#include <iostream>

#include "U4SensorIdentifier.h"
#include "G4PVPlacement.hh"

struct U4SensorIdentifierDefault : public U4SensorIdentifier 
{
    int getIdentity(const G4VPhysicalVolume* instance_outer_pv ) const ; 
    static void FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth );  
}; 


inline int U4SensorIdentifierDefault::getIdentity( const G4VPhysicalVolume* instance_outer_pv ) const 
{
    const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(instance_outer_pv) ;
    int copyno = pvp ? pvp->GetCopyNo() : -1 ;

    std::vector<const G4VPhysicalVolume*> sdpv ; 
    FindSD_r(sdpv, instance_outer_pv, 0 );  

    unsigned num_sd = sdpv.size() ; 
    int sensor_id = num_sd == 0 ? -1 : copyno ; 

    bool dump = copyno < 10 ; 
    if(dump) std::cout 
        << "U4SensorIdentifierDefault::getIdentity" 
        << " copyno " << copyno
        << " num_sd " << num_sd
        << " sensor_id " << sensor_id 
        << std::endl 
        ;      


    return sensor_id ; 
}

inline void U4SensorIdentifierDefault::FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth )
{
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    G4VSensitiveDetector* sd = lv->GetSensitiveDetector() ;
    if(sd) sdpv.push_back(pv); 
    for (size_t i=0 ; i < size_t(lv->GetNoDaughters()) ; i++ ) FindSD_r( sdpv, lv->GetDaughter(i), depth+1 );
}


