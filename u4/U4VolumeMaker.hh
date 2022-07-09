#pragma once

#include <string>
#include <vector>
class G4LogicalVolume ; 
class G4VPhysicalVolume ; 
class G4Material ; 

#include "G4ThreeVector.hh"
#include "U4_API_EXPORT.hh"
#include "plog/Severity.h"

struct U4_API U4VolumeMaker
{
    static const plog::Severity LEVEL ; 
    static const char* GEOM ; 
    static std::string Desc(); 
    static std::string Desc( const G4ThreeVector& tla ); 

    // top level interface : pv maker

    static G4VPhysicalVolume* PV(); // sensitive to GEOM envvar 
    static G4VPhysicalVolume* PV(  const char* name); 

    static G4VPhysicalVolume* PVG_(const char* name); 
    static G4VPhysicalVolume* PVP_(const char* name); 
    static G4VPhysicalVolume* PVS_(const char* name); 
    static G4VPhysicalVolume* PVL_(const char* name); 
    static G4VPhysicalVolume* PV1_(const char* name); 
  
 
    // general LV maker using U4SolidMaker::Make, U4Material::FindMaterialName
    static G4LogicalVolume*   LV(const char* name); 
    static void               LV(std::vector<G4LogicalVolume*>& lvs , const char* names_, char delim=',' ) ; 


    // WorldBox wrappers : arranging lv variously and returning world pv 

    static constexpr const char* U4VolumeMaker_WrapRockWater_HALFSIDE = "U4VolumeMaker_WrapRockWater_HALFSIDE" ; 
    static constexpr const char* U4VolumeMaker_WrapRockWater_FACTOR   = "U4VolumeMaker_WrapRockWater_FACTOR" ; 
    static G4VPhysicalVolume* WrapRockWater( G4LogicalVolume* lv ); 

    static constexpr const char* U4VolumeMaker_WrapVacuum_HALFSIDE = "U4VolumeMaker_WrapVacuum_HALFSIDE" ; 
    static G4VPhysicalVolume* WrapVacuum(   G4LogicalVolume* lv ); 


    static G4VPhysicalVolume* WrapLVGrid( G4LogicalVolume* lv, int nx, int ny, int nz  ); 
    static G4VPhysicalVolume* WrapLVGrid( std::vector<G4LogicalVolume*>& lvs, int nx, int ny, int nz  ); 
    static G4VPhysicalVolume* WrapLVOffset( G4LogicalVolume* lv, double tx, double ty, double tz ); 
    static G4VPhysicalVolume* WrapLVCube(   G4LogicalVolume* lv, double tx, double ty, double tz ); 
    static G4VPhysicalVolume* AddPlacement( G4LogicalVolume* mother, G4LogicalVolume* lv,  double tx, double ty, double tz ); 
    static G4VPhysicalVolume* AddPlacement( G4LogicalVolume* mother, const char* name   ,  double tx, double ty, double tz ); 
    static const char* GridName(const char* prefix, int ix, int iy, int iz, const char* suffix); 

    // world box pv

    static G4VPhysicalVolume* WorldBox( double halfside, const char* mat="Vacuum" ); 
    static G4VPhysicalVolume* BoxOfScintillator( double halfside ); 

    // box pv 

    static G4VPhysicalVolume* BoxOfScintillator( double halfside, const char* prefix, G4LogicalVolume* mother_lv ); 
    static G4VPhysicalVolume* Box(double halfside, const char* mat, const char* prefix, G4LogicalVolume* mother_lv ); 
    static G4VPhysicalVolume* Place( G4LogicalVolume* lv, G4LogicalVolume* mother_lv, const char* flip_axes=nullptr ); 

    // specialist pv creators

    static constexpr const char* U4VolumeMaker_RaindropRockAirWater_HALFSIDE = "U4VolumeMaker_RaindropRockAirWater_HALFSIDE" ; 
    static constexpr const char* U4VolumeMaker_RaindropRockAirWater_FACTOR   = "U4VolumeMaker_RaindropRockAirWater_FACTOR" ; 
    static void RaindropRockAirWater_Configure( double& rock_halfside, double& air_halfside, double& water_radius ); 
    static G4VPhysicalVolume* RaindropRockAirWater();  
    static G4VPhysicalVolume* RaindropRockAirWaterSD();

    // general lv creators 
 
    static G4LogicalVolume* Orb_( double radius,   const char* mat, const char* prefix=nullptr ); 
    static G4LogicalVolume* Box_( double halfside, const char* mat, const char* prefix=nullptr ); 
};


