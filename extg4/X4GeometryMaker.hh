#pragma once

#include <vector>
class G4VSolid ; 

#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"

struct X4_API X4GeometryMaker
{
    static const plog::Severity LEVEL ; 

    static bool  CanMake(const char* name); 
    static const G4VSolid* Make(const char* name); 

    static const G4VSolid* make_orb(const char* name); 
    static const G4VSolid* make_AdditionAcrylicConstruction(const char* name);
    static const G4VSolid* make_BoxMinusTubs0(const char* name);
    static const G4VSolid* make_BoxMinusTubs1(const char* name); 
    static const G4VSolid* make_UnionOfHemiEllipsoids(const char* name); 

    static void Extract( std::vector<long>& vals, const char* s ); 
    static bool StartsWith( const char* n, const char* q ); 

};


