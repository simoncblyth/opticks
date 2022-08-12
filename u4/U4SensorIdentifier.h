#pragma once
/**
U4SensorIdentifier.h
======================

Pure virtual protocol base used to interface detector
specific characteristics of sensors with Opticks. 

getIdentity
    method is called on the outer volume of every factorized instance during geometry translation
    If the subtree of volumes within the outer volume provided contains a sensor then 
    this method is expected to return an integer value that uniquely identifies the sensor. 
    If the subtree does not contain a sensor, then -1 should be returned. 


U4SensorIdentifierDefault.h provided the default implementation. 
To override this default use U4Tree::SetSensorIdentifier 

**/
class G4VPhysicalVolume ; 

struct U4SensorIdentifier
{
    virtual int getIdentity(const G4VPhysicalVolume* instance_outer_pv ) const = 0 ; 
}; 


