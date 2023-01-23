#pragma once
/**
U4Surface.h
==============

HMM distinction between border and skin can just be 
carried via the directory path and metadata ? 

HMM: maybe need to enhance NPFold.h metadata or could 
use a small array and plant metadata on that 

**/

#include "G4String.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4Version.hh"
#include "G4Material.hh"

#include "S4.h"

struct NPFold ; 


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


    static const std::vector<G4LogicalBorderSurface*>* PrepareBorderSurfaceVector(const G4LogicalBorderSurfaceTable* tab ); 


    static NPFold* MakeBorderFold(); 
    static NPFold* MakeBorderFold( const G4LogicalBorderSurfaceTable* tab); 

    static NPFold* MakeSkinFold(); 
    static NPFold* MakeSkinFold(const G4LogicalSkinSurfaceTable* tab); 

    static NPFold* MakeFold(); 

};


#include "U4Material.hh"
#include "U4MaterialPropertiesTable.h"
#include "U4Volume.h"
#include "NPFold.h"


inline unsigned U4Surface::Type(const char* type_)
{
    unsigned type = U4Surface_UNSET ; 
    if(strcmp(type_, PerfectAbsorber) == 0) type = U4Surface_PerfectAbsorber ; 
    if(strcmp(type_, PerfectDetector) == 0) type = U4Surface_PerfectDetector ; 
    return type ; 
}



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


/**
U4Surface::PrepareBorderSurfaceVector
---------------------------------------

Prior to Geant4 1070 G4LogicalBorderSurfaceTable was simply typedef to 
std::vector<G4LogicalBorderSurface*> (g4-cls G4LogicalBorderSurface)
for 1070 and above the table type changed to become a std::map with  
pair of pointers key : std::pair<const G4VPhysicalVolume*, const G4VPhysicalVolume*>.

As the std::map iteration order with such a key could potentially change from 
invokation to invokation or between platforms depending on where the pointer 
addresses got allocated it is necessary to impose a more meaningful 
and consistent order. 

As Opticks serializes all geometry objects into arrays for upload 
to GPU buffers and textures and uses indices to reference into these 
buffers and textures it is necessary for all collections of geometry objects 
to have well defined and consistent ordering.
To guarantee this the std::vector obtained from the std::map is sorted based on 
the 0x stripped name of the G4LogicalBorderSurface.

**/

inline const std::vector<G4LogicalBorderSurface*>* U4Surface::PrepareBorderSurfaceVector(const G4LogicalBorderSurfaceTable* tab )  // static
{
    typedef std::vector<G4LogicalBorderSurface*> VBS ; 
#if G4VERSION_NUMBER >= 1070
    typedef std::pair<const G4VPhysicalVolume*, const G4VPhysicalVolume*> PPV ; 
    typedef std::map<PPV, G4LogicalBorderSurface*>::const_iterator IT ; 

    VBS* vec = new VBS ;   
    for(IT it=tab->begin() ; it != tab->end() ; it++ )
    {   
        G4LogicalBorderSurface* bs = it->second ;    
        vec->push_back(bs);    
        const PPV ppv = it->first ; 
        assert( ppv.first == bs->GetVolume1());  
        assert( ppv.second == bs->GetVolume2());  
    }   

    {   
        bool reverse = false ; 
        const char* tail = "0x" ; 
        SNameOrder<G4LogicalBorderSurface>::Sort( *vec, reverse, tail ); 
        std::cout << SNameOrder<G4LogicalBorderSurface>::Desc( *vec ) << std::endl ; 
    }   

#else
    const VBS* vec = tab ;   
    // hmm maybe should name sort pre 1070 too for consistency 
    // otherwise they will stay in creation order
    // Do this once 107* becomes more relevant to Opticks.
#endif
    return vec ; 
}




inline NPFold* U4Surface::MakeBorderFold()  // static 
{
    const G4LogicalBorderSurfaceTable* tab = G4LogicalBorderSurface::GetSurfaceTable() ;
    return MakeBorderFold(tab); 
}

/**
U4Surface::MakeBorderFold
--------------------------

**/

inline NPFold* U4Surface::MakeBorderFold(const G4LogicalBorderSurfaceTable* tab) // static
{
    const std::vector<G4LogicalBorderSurface*>* vec = PrepareBorderSurfaceVector(tab); 
    if(vec == nullptr) return nullptr ; 

    NPFold* fold = new NPFold ; 

    for(unsigned i=0 ; i < vec->size() ; i++)
    {   
        const G4LogicalBorderSurface* bs = (*vec)[i] ; 
        const G4String& name = bs->GetName() ; 
        const char* key = name.c_str() ; 

        const G4VPhysicalVolume* _pv1 = bs->GetVolume1(); 
        const G4VPhysicalVolume* _pv2 = bs->GetVolume2(); 

        const char* pv1 = S4::Name<G4VPhysicalVolume>(_pv1) ;  // these names have 0x...
        const char* pv2 = S4::Name<G4VPhysicalVolume>(_pv2) ; 

        G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(bs->GetSurfaceProperty());
        G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
        assert(mpt); 

        NPFold* sub = U4MaterialPropertiesTable::MakeFold(mpt) ; 

        sub->set_meta<std::string>("pv1", pv1) ; 
        sub->set_meta<std::string>("pv2", pv2) ; 

        fold->add_subfold( key, sub );  
    }   
    return fold ; 
}



inline NPFold* U4Surface::MakeSkinFold() // static
{
    const G4LogicalSkinSurfaceTable* tab = G4LogicalSkinSurface::GetSurfaceTable() ; 
    return MakeSkinFold( tab ); 
}

inline NPFold* U4Surface::MakeSkinFold(const G4LogicalSkinSurfaceTable* tab)
{
    NPFold* fold = new NPFold ; 

    for(unsigned i=0 ; i < tab->size() ; i++)
    {   
        const G4LogicalSkinSurface* ks = (*tab)[i] ; 
        const G4String& name = ks->GetName() ; 
        const char* key = name.c_str() ; 

        G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(ks->GetSurfaceProperty());
        G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
        assert(mpt); 

        const G4LogicalVolume* _lv = ks->GetLogicalVolume();
        const char* lv = S4::Name<G4LogicalVolume>(_lv);   // name includes 0x...

        NPFold* sub = U4MaterialPropertiesTable::MakeFold(mpt) ; 
        sub->set_meta<std::string>("lv", lv ); 

        fold->add_subfold( key, sub );  
    }   
    return fold ; 
}
     
inline NPFold* U4Surface::MakeFold()
{
    NPFold* fold = new NPFold ; 
    fold->add_subfold("BorderSurface", MakeBorderFold() ); 
    fold->add_subfold("SkinSurface", MakeSkinFold() ); 
    return fold ; 
}

