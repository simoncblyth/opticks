#pragma once

#include <vector>
#include "scuda.h"
#include "squad.h"

#include "X4_API_EXPORT.hh"

struct quad4 ; 
struct quad6 ; 
struct float4 ; 
class G4VSolid ; 
struct NP ; 

struct X4_API X4Intersect
{
    X4Intersect( const G4VSolid* solid_ ); 
    void init(); 
    void scan(); 

    const G4VSolid* solid ; 
    const NP* gs ;
    float gridscale ;  
    bool dump ; 

    float4 ce ; 
    std::vector<int> cegs ; 
    std::vector<int> override_ce ;  
    std::vector<quad4> pp ;
    std::vector<quad4> ss ;
}; 

