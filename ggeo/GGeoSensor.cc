
#include <string>
#include "BStr.hh"

#include "GMaterial.hh"
#include "GGeo.hh"
#include "GGeoSensor.hh"
#include "GDomain.hh"
#include "GSkinSurface.hh"
#include "GOpticalSurface.hh"
#include "GSurfaceLib.hh"

#include "PLOG.hh"



/**
GGeoSensor::AddSensorSurfaces
------------------------------

See the similar AssimpGGeo::convertSensors from the old route
This is invoked by X4PhysicalVolume::convertSensors in direct route.

Hmm the material properties of the sensor are irrelevant currently, 
but the surface properties are relevant (see oxrap/cu/propagate.h) 
with 4 possibilities, with probabilities depending on the surface props:

1. SURFACE_ABSORB
2. SURFACE_DETECT
3. SURFACE_DREFLECT diffuse
4. SURFACE_SREFLECT specular  


**/

void GGeoSensor::AddSensorSurfaces( GGeo* gg )
{
    GMaterial* cathode_props = gg->getCathode() ; 
    if(!cathode_props)
    {
        LOG(fatal) << " require a cathode material to AddSensorSurfaces " ; 
        return ; 
    }

    unsigned nclv = gg->getNumCathodeLV();

    for(unsigned i=0 ; i < nclv ; i++)
    {
        const char* sslv = gg->getCathodeLV(i);
        unsigned index = gg->getNumMaterials() + gg->getNumSkinSurfaces() + gg->getNumBorderSurfaces() ;
        // standard materials/surfaces use the originating aiMaterial index, 
        // extend that for fake SensorSurface by toting up all 

        LOG(info) << "GGeoSensor::AddSensorSurfaces"
                  << " i " << i
                  << " sslv " << sslv
                  << " index " << index
                  ;

        GSkinSurface* gss = MakeSensorSurface(sslv, index);
        gss->setStandardDomain();  // default domain 
        gss->setSensor();
        gss->add(cathode_props); 

        LOG(info) << " gss " << gss->description();

        gg->add(gss);

        {
            // not setting sensor or domain : only the standardized need those
            GSkinSurface* gss_raw = MakeSensorSurface(sslv, index);
            gss_raw->add(cathode_props);
            gg->addRaw(gss_raw);
        }  
    }
}


/**
GGeoSensor::MakeSensorSurface
------------------------------

Originally from AssimpGGeo::convertSensors but relocated 
here to neutral GGeo territory for use from both the old Assimp 
route and the new direct route.


**/

GOpticalSurface* GGeoSensor::MakeOpticalSurface( const char* sslv )
{
    std::string name = BStr::trimPointerSuffixPrefix(sslv, NULL );
    name += GSurfaceLib::SENSOR_SURFACE ;

    LOG(fatal) 
           << " sslv " << sslv  
           << " name " << name
           ;  


    const char* osnam = name.c_str() ;
    const char* ostyp = "0" ;
    const char* osmod = "1" ;
    const char* osfin = "3" ;
    const char* osval = "1" ;

    // TODO: check effects of above adhoc choice of common type/model/finish/value 
    // TODO: add parse ctor that understands: "type=dielectric_dielectric;model=unified;finish=ground;value=1.0"

    GOpticalSurface* os = new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) ;
    return os ; 
}


GSkinSurface* GGeoSensor::MakeSensorSurface(const char* sslv, unsigned index ) // static 
{
    // standard materials/surfaces use the originating aiMaterial index, 
    // extend that for fake SensorSurface by toting up all 

    GOpticalSurface* os = MakeOpticalSurface( sslv ) ; 

    GSkinSurface* gss = new GSkinSurface(os->getName(), index, os);

    gss->setSkinSurface(sslv);


    // story continues in GBoundaryLib::standardizeSurfaceProperties
    // that no longer exists, now probably GSurfaceLib::getSensorSurface

    return gss ;   // NB are missing properties 
}




