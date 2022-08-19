#pragma once
/**
U4SensorIdentifier.h
======================

Pure virtual protocol base used to interface detector
specific characteristics of sensors with Opticks. 

getGlobalIdentity
    method is called on ALL of the remainder non-instanced nodes, 
    it is expected to return an integer value uniquely identifiying 
    any sensors. For non-sensor volumes an integer of -1 should be returned

    AS JUNO HAS NO global sensors this is untested. 

getInstanceIdentity
    method is called on ONLY the outer volume of every factorized 
    instance during geometry translation
    If the subtree of volumes within the outer volume provided 
    contains a sensor then this method is expected to return an 
    integer value that uniquely identifies the sensor. 
    If the subtree does not contain a sensor, then -1 should be returned. 

    CURRENTLY HAVING MORE THAN ONE ACTUAL SENSOR PER INSTANCE IS NOT SUPPORTED
    
    An ACTUAL sensor is one that would yield hits with Geant4 : ie it 
    must have an EFFICIENCY property with non-zero values and have 
    G4LogicalVolume::SetSensitiveDetector associated.  

U4SensorIdentifierDefault.h provided the default implementation. 
To override this default use U4Tree::SetSensorIdentifier 

**/
class G4VPhysicalVolume ; 

struct U4SensorIdentifier
{
    virtual int getGlobalIdentity(const G4VPhysicalVolume* node_pv ) const = 0 ; 
    virtual int getInstanceIdentity(const G4VPhysicalVolume* instance_outer_pv ) const = 0 ; 
}; 


