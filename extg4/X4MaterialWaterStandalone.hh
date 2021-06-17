#pragma once

#include "X4_API_EXPORT.hh"
#include <array>
#include "plog/Severity.h"

#include "G4Types.hh"

class G4Material ; 
class G4MaterialPropertiesTable ; 


/**
X4MaterialWaterStandalone
============================

Typically detector simulation frameworks define a G4Material called "Water"
however it is expedient to separately define such a material here in order 
to facilitate standalone testing. 

**/

struct X4_API X4MaterialWaterStandalone
{
    G4double                     density ;
    G4Material*                  Water ; 
    G4MaterialPropertiesTable*   WaterMPT ; 
    std::array<double, 36>       fPP_Water_RIN ; 
    std::array<double, 36>       fWaterRINDEX ; 

    X4MaterialWaterStandalone(); 
 
    void init(); 
    void initData(); 
    void dump() const ;
};


