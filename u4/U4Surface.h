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

#if G4VERSION_NUMBER >= 1070
#include "SNameOrder.h"
#endif



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

    static void    Collect( std::vector<const G4LogicalSurface*>& surfaces ); 
    static NPFold* MakeFold(const std::vector<const G4LogicalSurface*>& surfaces ); 
    static NPFold* MakeFold(); 

    static G4LogicalSurface* Find( const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV ) ;  

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



/**
U4Surface::Collect 
---------------------

Collects G4LogicalBorderSurface and G4LogicalSkinSurface pointers
into vector of G4LogicalSurface. 

**/


inline void U4Surface::Collect( std::vector<const G4LogicalSurface*>& surfaces )
{
    const G4LogicalBorderSurfaceTable* border_ = G4LogicalBorderSurface::GetSurfaceTable() ;
    const std::vector<G4LogicalBorderSurface*>* border = PrepareBorderSurfaceVector(border_); 

    for(unsigned i=0 ; i < border->size() ; i++)
    {   
        G4LogicalBorderSurface* bs = (*border)[i] ; 
        surfaces.push_back(bs) ;  
    }   

    const G4LogicalSkinSurfaceTable* skin = G4LogicalSkinSurface::GetSurfaceTable() ; 
    for(unsigned i=0 ; i < skin->size() ; i++)
    {   
        G4LogicalSkinSurface* ks = (*skin)[i] ; 
        surfaces.push_back(ks) ;  
    }
}

/**
U4Surface::MakeFold
--------------------

Canonical usage from U4Tree::initSurfaces creating the stree::surface NPFold. 

**/

inline NPFold* U4Surface::MakeFold(const std::vector<const G4LogicalSurface*>& surfaces ) // static
{
    NPFold* fold = new NPFold ; 
    for(unsigned i=0 ; i < surfaces.size() ; i++)
    {   
        const G4LogicalSurface* ls = surfaces[i] ; 
        const G4String& name = ls->GetName() ; 
        const char* key = name.c_str() ; 

        G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(ls->GetSurfaceProperty());

        G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;
        assert(mpt); 
        NPFold* sub = U4MaterialPropertiesTable::MakeFold(mpt) ; 

        const char* osn = os->GetName().c_str() ; 
        sub->set_meta<std::string>("OpticalSurfaceName", osn) ;  // ADDED for specials handling 

        const G4LogicalBorderSurface* bs = dynamic_cast<const G4LogicalBorderSurface*>(ls) ; 
        const G4LogicalSkinSurface*   ks = dynamic_cast<const G4LogicalSkinSurface*>(ls) ; 

        if(bs)
        {
            const G4VPhysicalVolume* _pv1 = bs->GetVolume1(); 
            const G4VPhysicalVolume* _pv2 = bs->GetVolume2(); 

            const char* pv1 = S4::Name<G4VPhysicalVolume>(_pv1) ;  // these names have 0x...
            const char* pv2 = S4::Name<G4VPhysicalVolume>(_pv2) ; 

            sub->set_meta<std::string>("pv1", pv1) ; 
            sub->set_meta<std::string>("pv2", pv2) ; 
            sub->set_meta<std::string>("type", "Border" ); 
        }
        else if(ks)
        {
            const G4LogicalVolume* _lv = ks->GetLogicalVolume();
            const char* lv = S4::Name<G4LogicalVolume>(_lv);   // name includes 0x...

            sub->set_meta<std::string>("lv", lv ); 
            sub->set_meta<std::string>("type", "Skin" ); 
        }
        fold->add_subfold( key, sub );  
    }   
    return fold ; 
}

inline NPFold* U4Surface::MakeFold()
{
    std::vector<const G4LogicalSurface*> surfaces ; 
    Collect(surfaces); 
    return MakeFold(surfaces) ; 
}

/**
U4Surface::Find
-----------------

Based on G4OpBoundaryProcess::PostStepDoIt 

**/

inline G4LogicalSurface* U4Surface::Find( const G4VPhysicalVolume* thePrePV, const G4VPhysicalVolume* thePostPV ) 
{
    if(thePostPV == nullptr || thePrePV == nullptr ) return nullptr ;  // surface on world volume not allowed 
    G4LogicalSurface* Surface = G4LogicalBorderSurface::GetSurface(thePrePV, thePostPV);
    if(Surface == nullptr)
    {
        G4bool enteredDaughter = thePostPV->GetMotherLogical() == thePrePV->GetLogicalVolume();
        if(enteredDaughter)
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
            if(Surface == nullptr)
                Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
        }    
        else 
        {
            Surface = G4LogicalSkinSurface::GetSurface(thePrePV->GetLogicalVolume());
            if(Surface == nullptr)
                Surface = G4LogicalSkinSurface::GetSurface(thePostPV->GetLogicalVolume());
        }    
    }    
    return Surface ; 
}

