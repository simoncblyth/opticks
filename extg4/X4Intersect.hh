#pragma once

#include <vector>
#include <string>
#include "scuda.h"
#include "squad.h"

#include "X4_API_EXPORT.hh"
#include "G4ThreeVector.hh"

struct quad4 ; 
struct quad6 ; 
struct float4 ; 
class G4VSolid ; 
struct NP ; 

struct X4_API X4Intersect
{
    static void Scan(const G4VSolid* solid, const char* name, const char* basedir, const std::string& meta ); 

    X4Intersect( const G4VSolid* solid_ ); 
    const char* desc() const ; 

    static double Distance(const G4VSolid* solid, const G4ThreeVector& pos, const G4ThreeVector& dir, bool dump); 

    void init(); 
    void scan(); 
    void save(const char* dir) const ;

    const G4VSolid* solid ; 
    NP* gs ;    // not const as need to externally set the meta 
    float gridscale ;  
    quad4* peta ; 
    bool dump ; 

    float4 ce ; 
    std::vector<int> cegs ; 
    std::vector<int> override_ce ;  
    std::vector<quad4> pp ;
    std::vector<quad4> ii ;
}; 

