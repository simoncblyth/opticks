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

#include "U4SurfaceType.h"
#include "U4OpticalSurfaceFinish.h"
#include "U4OpticalSurfaceModel.h"

#include "U4SurfacePerfect.h"
#include "S4.h"

#include "sdomain.h"
#include "sproplist.h"

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
    static const std::vector<G4LogicalSkinSurface*>*   PrepareSkinSurfaceVector(const G4LogicalSkinSurfaceTable* tab ); 

    static void    Collect( std::vector<const G4LogicalSurface*>& surfaces ); 
    static void    CollectRawNames( std::vector<std::string>& rawnames, const std::vector<const G4LogicalSurface*>& surfaces ); 

    static NPFold* MakeFold(const std::vector<const G4LogicalSurface*>& surfaces, const std::vector<std::string>& keys ); 
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


Note that prior to 1070 the table was a vector, and this preparation
does nothing so the order of the border surfaces is just the creation order. 
For consistency of the order between Geant4 versions the surfaces could be 
name sorted, but this is not yet done as there has been no need for consistent
surface indices between Geant4 versions.  

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
        std::cout << "U4Surface::PrepareBorderSurfaceVector\n" << SNameOrder<G4LogicalBorderSurface>::Desc( *vec ) << std::endl ; 
    }   

#else
    const VBS* vec = tab ;   
#endif
    return vec ; 
}

/**
U4Surface::PrepareSkinSurfaceVector
------------------------------------

TODO : update the G4VERSION_NUMBER branch guess ">= 1122" to the appropriate one at which the skin surface table 
was changed from a vector to a map

**/


inline const std::vector<G4LogicalSkinSurface*>* U4Surface::PrepareSkinSurfaceVector(const G4LogicalSkinSurfaceTable* tab )  // static
{
    typedef std::vector<G4LogicalSkinSurface*> VKS ; 
#if G4VERSION_NUMBER >= 1122
    typedef std::map<const G4LogicalVolume*,G4LogicalSkinSurface*>::const_iterator IT ; 
    VKS* vec = new VKS ;    // not const as need to push_back

    for(IT it=tab->begin() ; it != tab->end() ; it++ )
    {   
        G4LogicalSkinSurface* ks = it->second ;    
        vec->push_back(ks);    
    }   

    {   
        bool reverse = false ; 
        const char* tail = "0x" ; 
        SNameOrder<G4LogicalSkinSurface>::Sort( *vec, reverse, tail ); 
        std::cout << "U4Surface::PrepareSkinSurfaceVector\n" << SNameOrder<G4LogicalSkinSurface>::Desc( *vec ) << std::endl ; 
    }   

#else
    const VKS* vec = tab ;   
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

    const G4LogicalSkinSurfaceTable* skin_ = G4LogicalSkinSurface::GetSurfaceTable() ; 
    const std::vector<G4LogicalSkinSurface*>* skin = PrepareSkinSurfaceVector(skin_); 
    for(unsigned i=0 ; i < skin->size() ; i++)
    {   
        G4LogicalSkinSurface* ks = (*skin)[i] ; 
        surfaces.push_back(ks) ;  
    }
}

inline void U4Surface::CollectRawNames( std::vector<std::string>& rawnames, const std::vector<const G4LogicalSurface*>& surfaces )
{
    for(unsigned i=0 ; i < surfaces.size() ; i++)
    {   
        const G4LogicalSurface* ls = surfaces[i] ; 
        const G4String& name = ls->GetName() ; 
        const char* raw = name.c_str() ; 
        rawnames.push_back(raw); 
    }
}



/**
U4Surface::MakeFold
--------------------

Canonical usage from U4Tree::initSurfaces creating the stree::surface NPFold. 

**/

inline NPFold* U4Surface::MakeFold(const std::vector<const G4LogicalSurface*>& surfaces, const std::vector<std::string>& keys ) // static
{
    assert( surfaces.size() == keys.size() ); 

    NPFold* fold = new NPFold ; 
    for(unsigned i=0 ; i < surfaces.size() ; i++)
    {   
        const G4LogicalSurface* ls = surfaces[i] ; 
        [[maybe_unused]] const char* rawname = ls->GetName().c_str() ; 
        const char* key = keys[i].c_str() ; 

        G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(ls->GetSurfaceProperty());

        G4SurfaceType theType = os->GetType();
        G4OpticalSurfaceModel theModel = os->GetModel();
        G4OpticalSurfaceFinish theFinish = os->GetFinish();       

        // cf X4OpticalSurface::Convert
        G4double ModelValue = theModel == glisur ? os->GetPolish() : os->GetSigmaAlpha() ;
        assert( ModelValue >= 0. && ModelValue <= 1. );

        const char* osn = os->GetName().c_str() ; 
        G4MaterialPropertiesTable* mpt = os->GetMaterialPropertiesTable() ;

        assert(mpt);  
        NPFold* sub = U4MaterialPropertiesTable::MakeFold(mpt) ; 

        sub->set_meta<std::string>("rawname", rawname) ; 
        sub->set_meta<std::string>("OpticalSurfaceName", osn) ; 
        sub->set_meta<std::string>("TypeName", U4SurfaceType::Name(theType)) ; 
        sub->set_meta<std::string>("ModelName", U4OpticalSurfaceModel::Name(theModel)) ; 
        sub->set_meta<std::string>("FinishName", U4OpticalSurfaceFinish::Name(theFinish)) ; 

        sub->set_meta<int>("Type", theType) ; 
        sub->set_meta<int>("Model", theModel) ; 
        sub->set_meta<int>("Finish", theFinish) ; 
        sub->set_meta<double>("ModelValue", ModelValue ) ; 


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
    //assert(0) ; // this is just used from U4SurfaceTest it seems 
    std::vector<const G4LogicalSurface*> surfaces ; 
    Collect(surfaces); 

    std::vector<std::string> suname_raw ;  
    U4Surface::CollectRawNames(suname_raw, surfaces); 

    std::vector<std::string> suname ;  
    sstr::StripTail_Unique( suname, suname_raw, "0x" );

    return MakeFold(surfaces, suname) ; 
}



/**
U4Surface::Find
-----------------

Looks for a border or skin surface in the same way 
as G4OpBoundaryProcess::PostStepDoIt which the code
is based on. 

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



