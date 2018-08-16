#include <sstream>

#include "G4MaterialPropertiesTable.hh"
#include "G4OpticalSurface.hh"

#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalVolume.hh"

#include "BStr.hh"

#include "Opticks.hh"

#include "GVector.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GOpticalSurface.hh"
#include "GSurfaceLib.hh"

#include "CMPT.hh"
#include "CSurfaceLib.hh"
#include "COptical.hh"
#include "COpticalSurface.hh"
#include "CDetector.hh"

#include "PLOG.hh"


CSurfaceLib::CSurfaceLib(GSurfaceLib* surfacelib) 
    :
    m_surfacelib(surfacelib),
    m_ok(surfacelib->getOpticks()),
    m_dbgsurf(m_ok->isDbgSurf()),
    m_detector(NULL),
    m_level(info)
{
    LOG(m_level) << "." ; 
}

void CSurfaceLib::setDetector(CDetector* detector)
{
    assert(m_detector == NULL);
    m_detector = detector ; 
}

std::string CSurfaceLib::brief()
{
    std::stringstream ss; 

    ss << "CSurfaceLib " 
       << " numBorderSurface " << m_border.size() 
       << " numSkinSurface " << m_skin.size() 
       ;

    return ss.str();
}

/**
CSurfaceLib::convert(CDetector* detector)
------------------------------------------

Invoked from CDetector::attachSurfaces which is invoked
from both CGDMLDetector::init and CTestDetector::init.

Creates G4LogicalBorderSurface + G4LogicalSkinSurface and associated G4OpticalSurface
from GSurfaceLib instance, collecting them into m_border and m_skin vectors.

* CDetector is required by makeBorderSurface makeSkinSurface to access the G4 lv and pv 

TODO: see if can rearrange into an easier to grasp position, eg::

    detector->attachSurfaces(CSurfaceLib* ) 

**/




void CSurfaceLib::convert(CDetector* detector, bool exclude_sensors)
{
    LOG(m_level) << "." ;

    //assert(m_surfacelib->isClosed()); 
    if(!m_surfacelib->isClosed())
    {
        LOG(info) << "CSurfaceLib::convert closing surfacelib " ; 
        m_surfacelib->close();
    }

    setDetector(detector);  

    unsigned num_surf = m_surfacelib->getNumSurfaces() ; 
    LOG(m_level) << "." 
                 << " num_surf " << num_surf
                  ; 

    if(m_dbgsurf)
    LOG(info) << "[--dbgsurf] CSurfaceLib::convert  num_surf " << num_surf  ;   

    for(unsigned i=0 ; i < num_surf ; i++)
    {   
        GPropertyMap<float>* surf = m_surfacelib->getSurface(i);
        const char* name = surf->getName(); 
        bool is_sensor_surface = GSurfaceLib::NameEndsWithSensorSurface( name ) ; 

        if( is_sensor_surface && exclude_sensors )
        {
            LOG(error) << " skip sensor surf : " 
                       << " name " << name 
                       << " keys " << surf->getKeysString()
                       ; 
            continue ; 
        }

        if(surf->isBorderSurface()) 
        {
             G4OpticalSurface* os = makeOpticalSurface(surf);
             G4LogicalBorderSurface* lbs = makeBorderSurface(surf, os);
             m_border.push_back(lbs);
        }
        else if(surf->isSkinSurface())
        {
             G4OpticalSurface* os = makeOpticalSurface(surf);
             G4LogicalSkinSurface* lss = makeSkinSurface(surf, os);
             m_skin.push_back(lss);
        } 
        else
        {
             LOG(fatal) << " SKIPPED surface "
                        << " i "  << i
                        << " brief : "  << surf->brief() 
                        << " meta : "  << surf->getMetaDesc() 
                        ;  
        }
    }   
    LOG(info) << brief();
}


G4OpticalSurface* CSurfaceLib::makeOpticalSurface(GPropertyMap<float>* surf )
{
    GOpticalSurface* os_ = surf->getOpticalSurface();

    const char* name = os_->getName() ;
    guint4 optical   = os_->getOptical();


    G4OpticalSurface* os = new G4OpticalSurface(name);

    G4OpticalSurfaceModel model = COptical::Model(1) ; 
    G4OpticalSurfaceFinish finish = COptical::Finish(optical.z) ;  //  polished,,,ground,,, 
    G4SurfaceType type = COptical::Type(optical.y) ;   // dielectric_metal, dielectric_dielectric

    os->SetModel(model);
    os->SetFinish(finish);
    os->SetType(type);


    // fixed value omission : see notes/issues/ab-surf1.rst
    unsigned upercent = optical.w ; 
    float value = float(upercent)/100.f ; 

    LOG(verbose) 
        << " upercent (optical.w) " << upercent
        << " value " << value 
        ;


    if( model == glisur ) 
    {
        os->SetPolish(value) ; 
    } 
    else
    {
        os->SetSigmaAlpha(value) ; 
    }


    LOG(debug) << "CSurfaceLib::makeOpticalSurface " 
               << COpticalSurface::Brief(os) 
               ;

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    os->SetMaterialPropertiesTable(mpt);

    addProperties(mpt,  surf);

    return os ; 
}


// lookup G4 pv1,pv2 from volume indices recorded in pairs
G4LogicalBorderSurface* CSurfaceLib::makeBorderSurface(GPropertyMap<float>* surf, G4OpticalSurface* os)
{
    const char* name = surf->getName() ;

    //  queasy about using name lookup for PV
    //  see notes/issues/surface_review.rst#Potential_Missing_Surfaces

    std::string bpv1 = surf->getBPV1();
    std::string bpv2 = surf->getBPV2();

    bool trimPtr = false ; 
    char* pvn1 = BStr::DAEIdToG4(bpv1.c_str(), trimPtr);
    char* pvn2 = BStr::DAEIdToG4(bpv2.c_str(), trimPtr );


    if(m_dbgsurf)
    LOG(info) << "CSurfaceLib::makeBorderSurface"
              << " name " << name 
              << " bpv1 " << bpv1
              << " bpv2 " << bpv2
              << " pvn1 " << pvn1
              << " pvn2 " << pvn2
              ;

    const G4VPhysicalVolume* pv1 = m_detector->getPV(pvn1);    
    const G4VPhysicalVolume* pv2 = m_detector->getPV(pvn2);    

    assert( pv1 );
    assert( pv2 );     

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,
             const_cast<G4VPhysicalVolume*>(pv1),
             const_cast<G4VPhysicalVolume*>(pv2),
             os);

    return lbs ; 
}


// lookup G4 lv via name 
G4LogicalSkinSurface* CSurfaceLib::makeSkinSurface(GPropertyMap<float>* surf, G4OpticalSurface* os)
{
    const char* name = surf->getName() ;
    std::string sslv = surf->getSSLV() ;

    bool trimPtr = false ; 
    char* lvn = BStr::DAEIdToG4(sslv.c_str(), trimPtr);
    const G4LogicalVolume* lv = m_detector->getLV(lvn);

    if(m_dbgsurf)
    LOG(info) << "CSurfaceLib::makeSkinSurface"
              << " name " << std::setw(35) << name
              << " lvn " << std::setw(35) << lvn 
              << " lv " << ( lv ? lv->GetName() : "NULL" )
              ;

    assert(lv) ;

    G4LogicalSkinSurface* lss = new G4LogicalSkinSurface(name, const_cast<G4LogicalVolume*>(lv), os );
    return lss ;
}



void CSurfaceLib::addProperties(G4MaterialPropertiesTable* mpt_, GPropertyMap<float>* pmap)
{
    /**
    Property values hail from GSurfaceLib::createStandardSurface  
    which did the preparations for the Opticks texture, so 
    there should be little to do here other than translate into 
    EFFICIENCY and REFLECTIVITY ?
    **/

    CMPT mpt(mpt_);

    GOpticalSurface* os_ = pmap->getOpticalSurface();

    //unsigned   nprop = pmap->getNumProperties();
    //const char* name = pmap->getShortName();
    //LOG(info) << "CSurfaceLib::addProperties " << name ;  

    GProperty<float>* detect = pmap->getProperty(GSurfaceLib::detect);
    GProperty<float>* absorb = pmap->getProperty(GSurfaceLib::absorb);
    GProperty<float>* specular = pmap->getProperty(GSurfaceLib::reflect_specular);
    GProperty<float>* diffuse = pmap->getProperty(GSurfaceLib::reflect_diffuse);

    bool is_sensor = pmap->isSensor();   // ?? always false 
    bool is_specular = os_->isSpecular();

    bool detect_zero = detect->isZero();
    bool absorb_zero = absorb->isZero();
    bool specular_zero = specular->isZero();
    bool diffuse_zero = diffuse->isZero();

    bool spline = false ; 

    std::stringstream ss, tt ; 

    if(!detect_zero) ss << " sensor " ;
    mpt.addProperty("EFFICIENCY",   detect, spline );


    // Opticks distinguishes specular from diffuse by putting 
    // the REFLECTIVITY prop in either 
    // reflect_specular or reflect_diffuse slot 
    // so reverse that here...

    if(specular_zero && diffuse_zero )
    {
        ss << " zerorefl " ;
        mpt.addProperty("REFLECTIVITY", specular , spline );
    }
    else if(specular_zero && !diffuse_zero )
    {
        ss << " diffuse " ;
        mpt.addProperty("REFLECTIVITY", diffuse , spline );
    }
    else if(!specular_zero && diffuse_zero )
    {
        ss << " specular " ;
        mpt.addProperty("REFLECTIVITY", specular , spline );
    }
    else
    {
         assert(0);
    }

    if(detect_zero) tt << "detect_zero " ;
    if(absorb_zero) tt << "absorb_zero " ;
    if(specular_zero) tt << "specular_zero " ;
    if(diffuse_zero) tt << "diffuse_zero " ;

    if(is_sensor) tt << " is_sensor " ; 
    if(is_specular) tt << " is_specular " ; 

/*
    LOG(info) 
              << " name " << std::setw(35) << name
              << " nprop " << std::setw(4) << nprop
              << std::setw(30) << ss.str() 
              << std::setw(50) << tt.str() 
              ;

    LOG(info) << mpt.description("MPT:");

*/
}


