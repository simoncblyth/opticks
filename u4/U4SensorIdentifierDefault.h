#pragma once
/**
u4/U4SensorIdentifierDefault.h
================================

This fulfils U4SensorIdentifier protocol, it is used
to identify sensors in the geometry.  To override this
implementation use G4CXOpticks::SetSensorIdentifier.


**/

#include <vector>
#include <iostream>
#include <map>

#include "G4PVPlacement.hh"

#include "sstr.h"
#include "ssys.h"

#include "U4SensorIdentifier.h"
#include "U4Boundary.h"


struct U4SensorIdentifierDefault : public U4SensorIdentifier
{
    static std::vector<std::string>* GLOBAL_SENSOR_BOUNDARY_LIST ;

    void setLevel(int _level);
    int getGlobalIdentity(const G4VPhysicalVolume* pv, const G4VPhysicalVolume* ppv ) ;
    int getInstanceIdentity(const G4VPhysicalVolume* instance_outer_pv ) const ;
    static void FindSD_r( std::vector<const G4VPhysicalVolume*>& sdpv , const G4VPhysicalVolume* pv, int depth );
    static bool IsInterestingCopyNo( int copyno );

    int level = 0 ;
    std::vector<int> count_global_sensor_boundary ;

};


std::vector<std::string>*
U4SensorIdentifierDefault::GLOBAL_SENSOR_BOUNDARY_LIST = ssys::getenv_vec<std::string>("U4SensorIdentifierDefault__GLOBAL_SENSOR_BOUNDARY_LIST", "", '\n' );


inline void U4SensorIdentifierDefault::setLevel(int _level)
{
    level = _level ;
}

/**
U4SensorIdentifierDefault::getGlobalIdentity
---------------------------------------------

Canonically invoked from U4Tree::identifySensitiveGlobals

Currently a kludge using hardcoded pvn prefix.
This is because sensors within the global remainder
is only relevant to test geometries for JUNO.

Would be better to construct a boundary name and match that
against a list of sensor boundaries (see GBndLib__SENSOR_BOUNDARY_LIST)


**/

inline int U4SensorIdentifierDefault::getGlobalIdentity( const G4VPhysicalVolume* pv, const G4VPhysicalVolume* ppv )
{
    U4Boundary boundary(pv,ppv);

    const char* bnd = boundary.bnd.c_str() ;

    int id = ssys::listed_count( &count_global_sensor_boundary, GLOBAL_SENSOR_BOUNDARY_LIST, bnd ) ;

    if(level > 0) std::cout
        << "U4SensorIdentifierDefault::getGlobalIdentity "
        << " level " << level
        << " id " << id
        << " bnd " << bnd
        << std::endl
        ;


    return id ;
}


/**
U4SensorIdentifierDefault::getInstanceIdentity
---------------------------------------------------

Canonically used from U4Tree::identifySensitiveInstances

The argument *instance_outer_pv* is recursively traversed

Returns -1 to signify "not-a-sensor" otherwise returns
the copyno (aka "lpmtid") from the JUNO Geant4 PV of PMTs can be zero and
is used for identification of sensors with (unfortunately)
a non-contiguous set of values with some very large gaps.

HMM: if were to switch to using the contiguous "lpmtidx"
rather than the non-contiguous "lpmtid" this would likely
be the place to do it.  HMM: but its too JUNO specific for here ?

**/


inline int U4SensorIdentifierDefault::getInstanceIdentity( const G4VPhysicalVolume* instance_outer_pv ) const
{
    const char* pvn = instance_outer_pv ? instance_outer_pv->GetName().c_str() : "-" ;
    bool has_PMT_pvn = strstr(pvn, "PMT") != nullptr  ;

    const G4PVPlacement* pvp = dynamic_cast<const G4PVPlacement*>(instance_outer_pv) ;
    int copyno = pvp ? pvp->GetCopyNo() : -1 ;

    std::vector<const G4VPhysicalVolume*> sdpv ;
    FindSD_r(sdpv, instance_outer_pv, 0 );

    unsigned num_sd = sdpv.size() ;
    bool is_sensor = num_sd > 0 && has_PMT_pvn  ;

    int identifier = is_sensor ? copyno : -1  ;

    //bool is_interesting_copyno = IsInterestingCopyNo(copyno) ;
    //bool dump = is_sensor && is_interesting_copyno ;
    //bool dump = false ;
    //bool dump = true ;
    //bool dump = num_sd > 0 ;

    if(level > 0) std::cout
        << "U4SensorIdentifierDefault::getIdentity"
        << " level " << level
        << " copyno " << copyno
        << " num_sd " << num_sd
        << " is_sensor " << is_sensor
        << " pvn " << ( pvn ? pvn : "-" )
        << " has_PMT_pvn " << ( has_PMT_pvn ? "YES" : "NO " )
        << " identifier " << identifier
        << std::endl
        ;


    return identifier ;
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


inline bool U4SensorIdentifierDefault::IsInterestingCopyNo( int copyno )
{
    return
        copyno > -1 &&
           (
            (std::abs( copyno -      0 ) < 100) ||
            (std::abs( copyno -  17612 ) < 100) ||
            (std::abs( copyno -  30000 ) < 100) ||
            (std::abs( copyno -  32400 ) < 100) ||
            (std::abs( copyno - 300000 ) < 100) ||
            (std::abs( copyno - 325600 ) < 100)
           )
        ;
}


