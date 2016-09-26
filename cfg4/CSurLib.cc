#include <sstream>

#include "G4MaterialPropertiesTable.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"

#include "GVector.hh"
#include "GPropertyMap.hh"
#include "GOpticalSurface.hh"
#include "GSurfaceLib.hh"
#include "GSurLib.hh"
#include "GSur.hh"

#include "CMPT.hh"
#include "CSurLib.hh"
#include "CDetector.hh"

#include "PLOG.hh"



G4OpticalSurfaceModel Model(unsigned model_)
{
   // materials/include/G4OpticalSurface.hh
    G4OpticalSurfaceModel model = unified ;
    switch(model_)
    {
       case 0:model = glisur   ; break; 
       case 1:model = unified  ; break; 
       case 2:model = LUT      ; break; 
       case 3:model = dichroic ; break; 
    }
    return model ; 
}   
G4OpticalSurfaceFinish Finish(unsigned finish_)
{
   // materials/include/G4OpticalSurface.hh
    G4OpticalSurfaceFinish finish = polished ;
    switch(finish_)
    {
        case 0:finish = polished ;break;
        case 3:finish = ground   ;break;
    }
    return finish ; 
}
G4SurfaceType Type(unsigned type_)
{
    // materials/include/G4SurfaceProperty.hh
    G4SurfaceType type = dielectric_dielectric ;
    switch(type_)
    {
        case 0:type = dielectric_metal      ;break;
        case 1:type = dielectric_dielectric ;break;
    }
    return type ; 
}





CSurLib::CSurLib(GSurLib* surlib) 
   :
   m_surlib(surlib),
   m_surfacelib(surlib->getSurfaceLib()),
   m_detector(NULL)
{
}


void CSurLib::setDetector(CDetector* detector)
{
    assert(m_detector == NULL);
    m_detector = detector ; 
}


std::string CSurLib::brief()
{
    std::stringstream ss; 

    ss << "CSurLib " 
       << " num CSur " << m_surlib->getNumSur()
       << " --> "
       << " numBorderSurface " << m_border.size() 
       << " numSkinSurface " << m_skin.size() 
       ;

    return ss.str();
}


void CSurLib::convert(CDetector* detector)
{
    setDetector(detector);
    unsigned numSur = m_surlib->getNumSur();
    LOG(debug) << "CSurLib::convert  numSur " << numSur  ;   
    for(unsigned i=0 ; i < numSur ; i++)
    {   
        GSur* sur = m_surlib->getSur(i);
        G4OpticalSurface* os = makeOpticalSurface(sur);

        if(sur->isBorder()) 
        {
             unsigned nvp = sur->getNumVolumePair();
             for(unsigned ivp=0 ; ivp < nvp ; ivp++) 
             {
                 G4LogicalBorderSurface* lbs = makeBorderSurface(sur, ivp, os);
                 m_border.push_back(lbs);
             }
        }
        else if(sur->isSkin())
        {
             unsigned nlv = sur->getNumLV();
             for(unsigned ilv=0 ; ilv < nlv ; ilv++) 
             {
                 G4LogicalSkinSurface* lss = makeSkinSurface(sur, ilv, os);
                 m_skin.push_back(lss);
             }
        } 
    }   
    LOG(info) << brief();
}


G4OpticalSurface* CSurLib::makeOpticalSurface(GSur* sur)
{
    GPropertyMap<float>* pmap = sur->getPMap();
    GOpticalSurface* os_ = pmap->getOpticalSurface();
    G4OpticalSurface* os = new G4OpticalSurface(os_->getName());
    guint4 optical = os_->getOptical();

    os->SetModel(Model(1));
    os->SetFinish(Finish(optical.z));
    os->SetType(Type(optical.y));

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    os->SetMaterialPropertiesTable(mpt);

    addProperties(mpt,  pmap);

    return os ; 
}


G4LogicalBorderSurface* CSurLib::makeBorderSurface(GSur* sur, unsigned ivp, G4OpticalSurface* os)
{
    GPropertyMap<float>* pmap = sur->getPMap();
    const char* name = pmap->getName() ;
    guint4 pair = sur->getVolumePair(ivp);
    unsigned ipv1 = pair.x ; 
    unsigned ipv2 = pair.y ; 
    assert(pair.w == ivp);

    const G4VPhysicalVolume* pv1 = m_detector->getPV(ipv1);    
    const G4VPhysicalVolume* pv2 = m_detector->getPV(ipv2);    

    G4LogicalBorderSurface* lbs = new G4LogicalBorderSurface(name,
             const_cast<G4VPhysicalVolume*>(pv1),
             const_cast<G4VPhysicalVolume*>(pv2),
             os);

    return lbs ; 
}

G4LogicalSkinSurface* CSurLib::makeSkinSurface(GSur* sur, unsigned ilv, G4OpticalSurface* os)
{
    GPropertyMap<float>* pmap = sur->getPMap();
    const char* name = pmap->getName() ;
    const char* lvn = sur->getLV(ilv);       // assuming LV identity is 1-to-1 with name 
    const G4LogicalVolume* lv = m_detector->getLV(lvn);
    G4LogicalSkinSurface* lss = new G4LogicalSkinSurface(name, const_cast<G4LogicalVolume*>(lv), os );
    return lss ;
}



void CSurLib::addProperties(G4MaterialPropertiesTable* mpt_, GPropertyMap<float>* pmap)
{
    CMPT mpt(mpt_);

    const char* name = pmap->getShortName();
    unsigned   nprop = pmap->getNumProperties();

    LOG(info) << "CSurLib::addProperties"
              << " name " << std::setw(30) << name
              << " nprop " << std::setw(10) << nprop
              ;

    for(unsigned i=0 ; i<nprop ; i++)
    {
        const char* key =  pmap->getPropertyNameByIndex(i); 
        const char* lkey = m_surfacelib->getLocalKey(key) ;     
        bool spline = true ; 

        LOG(info) 
                  << std::setw(4) << i
                  << std::setw(30) << key 
                  << std::setw(30) << lkey 
                  ;

        GProperty<float>* prop = pmap->getPropertyByIndex(i);
        mpt.addProperty(lkey, prop, spline );
    }
}




