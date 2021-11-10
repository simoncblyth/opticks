#pragma once

class G4VSolid ; 
#include "GEOCHAIN_API_EXPORT.hh"
#include "plog/Severity.h"

struct GEOCHAIN_API GeoMaker
{
    static const plog::Severity LEVEL ; 

    static const G4VSolid* Make(const char* name); 
    static const G4VSolid* make_default(const char* name); 
    static const G4VSolid* make_AdditionAcrylicConstruction(const char* name);
    static const G4VSolid* make_BoxMinusTubs0(const char* name);
    static const G4VSolid* make_BoxMinusTubs1(const char* name); 
};



