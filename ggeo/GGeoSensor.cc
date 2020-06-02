/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


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


const plog::Severity GGeoSensor::LEVEL = PLOG::EnvLevel("GGeoSensor", "DEBUG")  ;



/**
GGeoSensor::AddSensorSurfaces
------------------------------

See the similar AssimpGGeo::convertSensors from the old route
This is invoked by X4PhysicalVolume::convertSensors in direct route.

This springs into life GGeo GSkinSurface/GOpticalSurface
with the material properties of the cathode material. 
This is done so sensitivity can survive the transition betweem
models.

Hmm the material properties of the sensor are irrelevant currently, 
but the surface properties are relevant (see oxrap/cu/propagate.h) 
with 4 possibilities, with probabilities depending on the surface props:

1. SURFACE_ABSORB
2. SURFACE_DETECT
3. SURFACE_DREFLECT diffuse
4. SURFACE_SREFLECT specular  


Issues/TODO
~~~~~~~~~~~~~

Currently assumes there is a single "cathode" material the properties
of which are assigned as SkinSurfaces to all logical volumes returned 
from GGeo::getNumCathodeLV GGeo::getCathodeLV.   

The upshot of this is that photons that succeed to reach a boundary
in optixrap/cu/generate.cu:generate will find surface properties associated
with the boundary resulting in the branching to propagate.h:propagate_at_surface 

Which means that one of the above SURFACE flags will be set. 
If SURFACE_DETECT gets set the photons will be copied back as hits.

Notice issues:

1. no handling of pre-existing surface assigned to the LV 
2. assumes a single cathode material (eg Bialkali for PMTs) with 
   a non-zero EFFICIENCY property : that gets adopted by the potentially multiple 
   lv with an associated SD 


**/

void GGeoSensor::AddSensorSurfaces( GGeo* gg )
{
#ifdef OLD_CATHODE
    GMaterial* cathode_props = gg->getCathode() ; 
    if(!cathode_props)
    {
        LOG(fatal) << " require a cathode material to AddSensorSurfaces " ; 
        return ; 
    }
    assert( cathode_props ); 
#else
    GMaterial* cathode_props = NULL ; 
#endif

    typedef std::vector<std::string> VS ;  
    VS lvn ; 
    VS sdn ; 
    VS mtn ; 

    gg->getSensitiveLVSDMT(lvn, sdn, mtn); 

    assert( lvn.size() == sdn.size() ) ;
    assert( lvn.size() == mtn.size() ) ;


    unsigned nclv = gg->getNumCathodeLV();

    if(nclv == 0)
    {
        LOG(error) << "NO CathodeLV : so not adding any GSkinSurface to translate sensitivity between models " ; 
    }

    assert( nclv == lvn.size() ) ; 



    for(unsigned i=0 ; i < nclv ; i++)
    {
        const char* lv = lvn[i].c_str();    
        const char* sd = sdn[i].c_str();    
        const char* mt = mtn[i].c_str();    

        GPropertyMap<float>* mt_props = gg->findMaterial(mt);
        assert( mt_props ); 

        const char* sslv = gg->getCathodeLV(i);
        assert( strcmp(lv, sslv) == 0 );    

        unsigned num_mat = gg->getNumMaterials()  ;
        unsigned num_sks = gg->getNumSkinSurfaces() ;
        unsigned num_bds = gg->getNumBorderSurfaces() ;

        unsigned index = num_mat + num_sks + num_bds ;
        // standard materials/surfaces use the originating aiMaterial index, 
        // extend that for fake SensorSurface by toting up all 

        LOG(LEVEL)
                  << " i " << i
                  << " sslv " << sslv
                  << " sd " << sd
                  << " mt " << mt
                  << " index " << index
                  << " num_mat " << num_mat
                  << " num_sks " << num_sks
                  << " num_bds " << num_bds
                  ;

        GSkinSurface* gss = MakeSensorSurface(sslv, index);
        gss->setStandardDomain();  // default domain 
        gss->setSensor();
#ifdef OLD_CATHODE
        gss->add(cathode_props); 
#else
        gss->add(mt_props);  
#endif

        LOG(LEVEL) << " gss " << gss->description();

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

    LOG(LEVEL) 
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




