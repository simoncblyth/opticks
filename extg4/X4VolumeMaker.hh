#pragma once

class G4LogicalVolume ; 
class G4VPhysicalVolume ; 

#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"

struct X4_API X4VolumeMaker
{
    static const plog::Severity LEVEL ; 
    
    static G4VPhysicalVolume* Make(const char* name); 

    static G4VPhysicalVolume* MakePhysical(const char* name); 
    static G4LogicalVolume*   MakeLogical(const char* name); 
    static G4VPhysicalVolume* WrapLVTranslate( G4LogicalVolume* lv, double tx, double ty, double tz ); 
    static G4VPhysicalVolume* WrapLVGrid( G4LogicalVolume* lv, int nx, int ny, int nz  ); 
    static G4VPhysicalVolume* WorldBox( double halfside ); 

    static const char* GridName(const char* prefix, int ix, int iy, int iz, const char* suffix); 

};


