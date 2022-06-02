#pragma once

#include <string>
#include <vector>
class G4LogicalVolume ; 
class G4VPhysicalVolume ; 

#include "G4ThreeVector.hh"
#include "U4_API_EXPORT.hh"
#include "plog/Severity.h"

struct U4_API U4VolumeMaker
{
    static const plog::Severity LEVEL ; 
    
    static G4VPhysicalVolume* Make(const char* name); 

    static G4VPhysicalVolume* MakePhysical(const char* name); 
    static G4VPhysicalVolume* MakePhysicalList_(const char* name); 
    static G4VPhysicalVolume* MakePhysicalOne_(const char* name); 

    static G4LogicalVolume*   MakeLogical(const char* name); 
    static void MakeLogical(std::vector<G4LogicalVolume*>& lvs , const char* names_ ) ; 

    static void AddPlacement( G4LogicalVolume* mother, G4LogicalVolume* lv,  double tx, double ty, double tz ); 
    static void AddPlacement( G4LogicalVolume* mother, const char* name   ,  double tx, double ty, double tz ); 

    static G4VPhysicalVolume* WrapLVOffset(    G4LogicalVolume* lv, double tx, double ty, double tz ); 
    static G4VPhysicalVolume* WrapLVTranslate( G4LogicalVolume* lv, double tx, double ty, double tz ); 

    static G4VPhysicalVolume* WrapLVGrid( G4LogicalVolume* lv, int nx, int ny, int nz  ); 
    static G4VPhysicalVolume* WrapLVGrid( std::vector<G4LogicalVolume*>& lvs, int nx, int ny, int nz  ); 

    static G4VPhysicalVolume* WorldBox( double halfside ); 
    static G4VPhysicalVolume* Wrap( G4LogicalVolume* lv, double halfside  ); 

    static std::string Desc( const G4ThreeVector& tla ); 

    static const char* GridName(const char* prefix, int ix, int iy, int iz, const char* suffix); 

};


