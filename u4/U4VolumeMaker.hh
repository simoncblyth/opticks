#pragma once

#include <string>
#include <vector>
class G4LogicalVolume ; 
class G4VPhysicalVolume ; 
class G4Material ; 
//class junoPMTOpticalModel ;

struct NP ; 

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

    static const G4VPhysicalVolume* PV(); // sensitive to GEOM envvar 
    static const G4VPhysicalVolume* PV(  const char* name); 

    static const char* PVG_WriteNames ; 
    static const char* PVG_WriteNames_Sub ; 

    static const G4VPhysicalVolume* PVG_(const char* name); 
    static const G4VPhysicalVolume* PVP_(const char* name); 
    static const G4VPhysicalVolume* PVF_(const char* name); 
    //static       junoPMTOpticalModel* PVF_POM ;  // set by last PVF_ call giving non-null pom

    static const G4VPhysicalVolume* PVS_(const char* name); 
    static const G4VPhysicalVolume* PVL_(const char* name); 
    static const G4VPhysicalVolume* PV1_(const char* name); 

  
 
    // general LV maker using U4SolidMaker::Make, U4Material::FindMaterialName
    static G4LogicalVolume*   LV(const char* name); 
    static void               LV(std::vector<G4LogicalVolume*>& lvs , const char* names_, char delim=',' ) ; 


    // WorldBox wrappers : arranging lv variously and returning world pv 

    static constexpr const char* U4VolumeMaker_WrapRockWater_HALFSIDE = "U4VolumeMaker_WrapRockWater_HALFSIDE" ; 
    static constexpr const char* U4VolumeMaker_WrapRockWater_FACTOR   = "U4VolumeMaker_WrapRockWater_FACTOR" ; 
    static constexpr const char* U4VolumeMaker_WrapRockWater_BS1 = "U4VolumeMaker_WrapRockWater_BS1" ; 

    static const G4VPhysicalVolume* Wrap( const char* name, std::vector<G4LogicalVolume*>& items_lv ); 
    static const G4VPhysicalVolume* WrapRockWater( std::vector<G4LogicalVolume*>& items_lv ); 

    static constexpr const char* U4VolumeMaker_WrapAroundItem_Rock_HALFSIDE = "U4VolumeMaker_WrapAroundItem_Rock_HALFSIDE" ; 
    static constexpr const char* U4VolumeMaker_WrapAroundItem_Water_HALFSIDE = "U4VolumeMaker_WrapAroundItem_Water_HALFSIDE" ; 
    static constexpr const char* U4VolumeMaker_WrapAroundItem_Rock_BOXSCALE = "U4VolumeMaker_WrapAroundItem_Rock_BOXSCALE" ; 
    static constexpr const char* U4VolumeMaker_WrapAroundItem_Water_BOXSCALE = "U4VolumeMaker_WrapAroundItem_Water_BOXSCALE" ; 
    static const G4VPhysicalVolume* WrapAroundItem( const char* name, std::vector<G4LogicalVolume*>& items_lv, const char* prefix ); // prefix eg AroundCircle


    static constexpr const char* U4VolumeMaker_WrapVacuum_HALFSIDE = "U4VolumeMaker_WrapVacuum_HALFSIDE" ; 
    static const G4VPhysicalVolume* WrapVacuum(   G4LogicalVolume* lv ); 


    static const G4VPhysicalVolume* WrapLVGrid( G4LogicalVolume* lv, int nx, int ny, int nz  ); 
    static const G4VPhysicalVolume* WrapLVGrid( std::vector<G4LogicalVolume*>& lvs, int nx, int ny, int nz  ); 
    static const G4VPhysicalVolume* WrapLVOffset( G4LogicalVolume* lv, double tx, double ty, double tz ); 
    static const G4VPhysicalVolume* WrapLVCube(   G4LogicalVolume* lv, double tx, double ty, double tz ); 
    static const G4VPhysicalVolume* AddPlacement( G4LogicalVolume* mother, G4LogicalVolume* lv,  double tx, double ty, double tz ); 
    static const G4VPhysicalVolume* AddPlacement( G4LogicalVolume* mother, const char* name   ,  double tx, double ty, double tz ); 
    static const char* GridName(const char* prefix, int ix, int iy, int iz, const char* suffix); 
    static const char* PlaceName(const char* prefix, int ix, const char* suffix); 

    // world box pv

    static const G4VPhysicalVolume* WorldBox( double halfside, const char* mat="Vacuum" ); 
    static const G4VPhysicalVolume* BoxOfScintillator( double halfside ); 

    // box pv 

    static const G4VPhysicalVolume* BoxOfScintillator( double halfside, const char* prefix, G4LogicalVolume* mother_lv ); 
    static const G4VPhysicalVolume* Box(double halfside, const char* mat, const char* prefix, G4LogicalVolume* mother_lv ); 
    static const G4VPhysicalVolume* Place( G4LogicalVolume* lv, G4LogicalVolume* mother_lv, const char* flip_axes=nullptr ); 

    // specialist pv creators

    static constexpr const char* U4VolumeMaker_RaindropRockAirWater_HALFSIDE = "U4VolumeMaker_RaindropRockAirWater_HALFSIDE" ; 
    static constexpr const char* U4VolumeMaker_RaindropRockAirWater_FACTOR   = "U4VolumeMaker_RaindropRockAirWater_FACTOR" ; 
    static void RaindropRockAirWater_Configure( double& rock_halfside, double& air_halfside, double& water_radius ); 
    static const G4VPhysicalVolume* RaindropRockAirWater();  
    static const G4VPhysicalVolume* RaindropRockAirWaterSD();

    // general lv creators 
 
    static G4LogicalVolume* Orb_( double radius,   const char* mat, const char* prefix=nullptr ); 
    static G4LogicalVolume* Box_( double halfside, const char* mat, const char* prefix=nullptr, const double* boxscale=nullptr ); 


    // instance wraps 
    static constexpr const char* U4VolumeMaker_MakeTransforms_AroundCircle_radius = "U4VolumeMaker_MakeTransforms_AroundCircle_radius" ; 
    static constexpr const char* U4VolumeMaker_MakeTransforms_AroundCircle_numInRing = "U4VolumeMaker_MakeTransforms_AroundCircle_numInRing" ; 
    static constexpr const char* U4VolumeMaker_MakeTransforms_AroundCircle_fracPhase = "U4VolumeMaker_MakeTransforms_AroundCircle_fracPhase" ; 
    static const NP* MakeTransforms( const char* name, const char* prefix ); 
    static void WrapAround( const char* prefix, const NP* trs, std::vector<G4LogicalVolume*>& lvs, G4LogicalVolume* mother_lv ); 

};


