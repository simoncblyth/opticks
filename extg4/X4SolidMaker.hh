#pragma once

#include <vector>
class G4VSolid ; 

#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"

struct X4_API X4SolidMaker
{
    static const plog::Severity LEVEL ; 
    static const char* NAMES ; 

    static bool  CanMake(const char* name); 
    static const G4VSolid* Make(const char* name); 

    static const G4VSolid* Orb(const char* name); 
    static const G4VSolid* SphereWithPhiSegment(const char* name); 
    static const G4VSolid* SphereWithThetaSegment(const char* name); 
    static const G4VSolid* AdditionAcrylicConstruction(const char* name);
    static const G4VSolid* BoxMinusTubs0(const char* name);
    static const G4VSolid* BoxMinusTubs1(const char* name); 
    static const G4VSolid* BoxMinusOrb(const char* name); 
    static const G4VSolid* UnionOfHemiEllipsoids(const char* name); 
    static const G4VSolid* PolyconeWithMultipleRmin(const char* name);

    static void Extract( std::vector<long>& vals, const char* s ); 
    static bool StartsWith( const char* n, const char* q ); 

};


