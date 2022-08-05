#pragma once

class G4LogicalBorderSurface ;
class G4OpticalSurface ; 
class G4VPhysicalVolume ; 
class G4LogicalSurface ; 

enum {
   U4Surface_UNSET, 
   U4Surface_PerfectAbsorber,
   U4Surface_PerfectDetector
};

struct U4Surface
{
    static constexpr const char* PerfectAbsorber = "PerfectAbsorber" ;
    static constexpr const char* PerfectDetector = "PerfectDetector" ;
    static unsigned Type(const char* type_); 

    static G4OpticalSurface* MakeOpticalSurface( const char* name_ ); 

    static G4LogicalBorderSurface* MakeBorderSurface(const char* name_, const char* type_, const char* pv1_, const char* pv2_, const G4VPhysicalVolume* start_pv ); 
    static G4LogicalBorderSurface* MakePerfectAbsorberBorderSurface(const char* name_, const char* pv1, const char* pv2, const G4VPhysicalVolume* start_pv  ); 
    static G4LogicalBorderSurface* MakePerfectDetectorBorderSurface(const char* name_, const char* pv1, const char* pv2, const G4VPhysicalVolume* start_pv  ); 

    static G4LogicalBorderSurface* MakeBorderSurface(const char* name_, const char* type_, const G4VPhysicalVolume* pv1, const G4VPhysicalVolume* pv2 ); 
    static G4LogicalBorderSurface* MakePerfectAbsorberBorderSurface(const char* name_, const G4VPhysicalVolume* pv1, const G4VPhysicalVolume* pv2 ); 
    static G4LogicalBorderSurface* MakePerfectDetectorBorderSurface(const char* name_, const G4VPhysicalVolume* pv1, const G4VPhysicalVolume* pv2 ); 

};

inline unsigned U4Surface::Type(const char* type_)
{
    unsigned type = U4Surface_UNSET ; 
    if(strcmp(type_, PerfectAbsorber) == 0) type = U4Surface_PerfectAbsorber ; 
    if(strcmp(type_, PerfectDetector) == 0) type = U4Surface_PerfectDetector ; 
    return type ; 
}

#include "G4String.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"

#include "U4Material.hh"
#include "U4Volume.h"



inline G4OpticalSurface* U4Surface::MakeOpticalSurface( const char* name_ )
{
    G4String name = name_ ; 
    G4OpticalSurfaceModel model = glisur ; 
    G4OpticalSurfaceFinish finish = polished ; 
    G4SurfaceType type = dielectric_dielectric ; 
    G4double value = 1.0 ; 
    G4OpticalSurface* os = new G4OpticalSurface(name, model, finish, type, value );  
    return os ; 
}

/**
U4Surface::MakeBorderSurface
--------------------------------------

From InstrumentedG4OpBoundaryProcess I think it needs a RINDEX property even though that is not 
going to be used for anything.  Also it needs REFLECTIVITY of zero. 

Getting G4OpBoundaryProcess to always give boundary status Detection for a surface requires:

1. REFLECTIVITY 0. forcing DoAbsoption 
2. EFFICIENCY 1. forcing Detection 

**/


inline G4LogicalBorderSurface* U4Surface::MakeBorderSurface(const char* name_, const char* type_, const char* pv1_, const char* pv2_, const G4VPhysicalVolume* start_pv )
{
    const G4VPhysicalVolume* pv1 = U4Volume::FindPV( start_pv, pv1_ ); 
    const G4VPhysicalVolume* pv2 = U4Volume::FindPV( start_pv, pv2_ ); 
    return ( pv1 && pv2 ) ? MakeBorderSurface(name_, type_, pv1, pv2 ) : nullptr ;  
}

inline G4LogicalBorderSurface* U4Surface::MakePerfectAbsorberBorderSurface(const char* name_, const char* pv1, const char* pv2, const G4VPhysicalVolume* start_pv)
{
    return MakeBorderSurface(name_, PerfectAbsorber, pv1, pv2, start_pv ); 
}
inline G4LogicalBorderSurface* U4Surface::MakePerfectDetectorBorderSurface(const char* name_, const char* pv1, const char* pv2, const G4VPhysicalVolume* start_pv)
{
    return MakeBorderSurface(name_, PerfectDetector, pv1, pv2, start_pv ); 
}




inline G4LogicalBorderSurface* U4Surface::MakeBorderSurface(const char* name_, const char* type_, const G4VPhysicalVolume* pv1, const G4VPhysicalVolume* pv2)
{
    unsigned type = Type(type_); 

    G4OpticalSurface* os = MakeOpticalSurface( name_ );  
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable ; 
    os->SetMaterialPropertiesTable(mpt);  

    G4MaterialPropertyVector* rindex = U4Material::MakeProperty(1.);  
    mpt->AddProperty("RINDEX", rindex );  

    G4MaterialPropertyVector* reflectivity = U4Material::MakeProperty(0.);  
    mpt->AddProperty("REFLECTIVITY",reflectivity );  


    if( type == U4Surface_PerfectAbsorber )
    {  
    }
    else if(  type == U4Surface_PerfectDetector )
    {
        G4MaterialPropertyVector* efficiency = U4Material::MakeProperty(1.);  
        mpt->AddProperty("EFFICIENCY",efficiency );  
    }

    G4String name = name_ ; 

    G4VPhysicalVolume* pv1_ = const_cast<G4VPhysicalVolume*>(pv1); 
    G4VPhysicalVolume* pv2_ = const_cast<G4VPhysicalVolume*>(pv2); 
    G4LogicalBorderSurface* bs = new G4LogicalBorderSurface(name, pv1_, pv2_, os ); 
    return bs ; 
}

inline G4LogicalBorderSurface* U4Surface::MakePerfectAbsorberBorderSurface(const char* name_, const G4VPhysicalVolume* pv1, const G4VPhysicalVolume* pv2)
{
    return MakeBorderSurface(name_, PerfectAbsorber, pv1, pv2 ); 
}
inline G4LogicalBorderSurface* U4Surface::MakePerfectDetectorBorderSurface(const char* name_, const G4VPhysicalVolume* pv1, const G4VPhysicalVolume* pv2)
{
    return MakeBorderSurface(name_, PerfectDetector, pv1, pv2 ); 
}


