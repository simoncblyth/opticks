#pragma once

#include <vector>
#include <string>
#include "X4_API_EXPORT.hh"
#include "G4ThreeVector.hh"
#include "geomdefs.hh"

struct float4 ; 
struct SCenterExtentGenstep ; 
class G4VSolid ; 
class G4MultiUnion ;

struct X4_API X4Intersect
{
    static void Scan(const G4VSolid* solid, const char* name, const char* basedir ); 

    X4Intersect( const G4VSolid* solid_ ); 
    const char* desc() const ; 

    static double Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump); 
    static double Distance_(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in  ); 
    static double DistanceMultiUnionNoVoxels_(const G4MultiUnion* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, EInside& in );

    void init(); 
    void scan_(); 
    void scan(); 

    const G4VSolid* solid ; 
    float4*         ce ; 
    SCenterExtentGenstep* cegs ; 
}; 

