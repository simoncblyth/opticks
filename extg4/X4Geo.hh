#pragma once
#include "X4_API_EXPORT.hh"

#include "plog/Severity.h"

class GGeo ; 
class G4VPhysicalVolume ; 

struct X4_API X4Geo
{
    static const plog::Severity LEVEL ; 
    static GGeo* Translate(const G4VPhysicalVolume* top); 
}; 

