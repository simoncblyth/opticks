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

// TEST=OKX4Test om-t

#include "SSys.hh"

#include "Opticks.hh"     
#include "OpticksQuery.hh"
#include "OpticksCfg.hh"

#include "OpticksHub.hh" 
#include "OpticksIdx.hh" 
#include "OpticksViz.hh"
#include "OKPropagator.hh"

#include "OKMgr.hh"     

#include "GBndLib.hh"
#include "GGeo.hh"
#include "GGeoGLTF.hh"

#include "CGDML.hh"
#include "CGDMLDetector.hh"

#include "X4PhysicalVolume.hh"
#include "X4Sample.hh"

class G4VPhysicalVolume ;

#include "G4PVPlacement.hh"
#include "OPTICKS_LOG.hh"

/**
OKX4Test : checking direct from G4 conversion, starting from a GDML loaded geometry
=======================================================================================

See :doc:`../../notes/issues/OKX4Test`

TODO: too much duplication between here and G4Opticks, perhaps can make 
      a G4OpticksMgr used by both ?
**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    for(int i=0 ; i < argc ; i++)  LOG(info) << i << " " << argv[i] ; 

    const char* gdmlpath = PLOG::instance->get_arg_after("--gdmlpath", NULL) ; 

    const char* csgskiplv = PLOG::instance->get_arg_after("--csgskiplv", NULL) ; 
    LOG(info) << " csgskiplv " << ( csgskiplv ? csgskiplv : "NONE" ) ;  

    const char* digestextra2 = PLOG::instance->get_arg_after("--digestextra", NULL) ; 
    LOG(info) << " digestextra2 " << ( digestextra2 ? digestextra2 : "NONE" ) ;  

    G4VPhysicalVolume* top = CGDML::Parse( gdmlpath ) ; 
    if( top == NULL ) return 0 ; 

    const char* digestextra1 = csgskiplv ;    // kludge the digest to be sensitive to csgskiplv
    const char* spec = X4PhysicalVolume::Key(top, digestextra1, digestextra2 ) ; 

    Opticks::SetKey(spec);

    const char* argforce = "--tracer --nogeocache --xanalytic" ;   // --nogeoache to prevent GGeo booting from cache 

    Opticks* m_ok = new Opticks(argc, argv, argforce);  // Opticks instanciation must be after Opticks::SetKey
    m_ok->configure();
    m_ok->enforceNoGeoCache(); 

    m_ok->profile("_OKX4Test:GGeo"); 

    GGeo* m_ggeo = new GGeo(m_ok) ;

    m_ok->profile("OKX4Test:GGeo"); 


    m_ok->profile("_OKX4Test:X4PhysicalVolume"); 

    X4PhysicalVolume xtop(m_ggeo, top) ;    // populates m_ggeo

    m_ok->profile("OKX4Test:X4PhysicalVolume"); 


    m_ggeo->postDirectTranslation();   // closing libs, finding repeat instances, merging meshes, saving 

    if(m_ok->isDumpSensor())
    {
        X4PhysicalVolume::DumpSensorVolumes(m_ggeo, "OKX4Test::main" ); 
        bool outer_volume = true ; 
        X4PhysicalVolume::DumpSensorPlacements(m_ggeo, "OKX4Test::main", outer_volume); 
    }

    m_ok->profile("_OKX4Test:OKMgr"); 
    assert( GGeo::GetInstance() == m_ggeo );

    // not OKG4Mgr as no need for CG4 
    OKMgr mgr(argc, argv);  // OpticksHub inside here picks up the gg (last GGeo instanciated) via GGeo::GetInstance 
    m_ok->profile("OKX4Test:OKMgr"); 
    //mgr.propagate();
    mgr.visualize();   

    Opticks* oki = Opticks::GetInstance() ; 
    assert( oki == m_ok ) ; 

    m_ok->saveProfile();
    m_ok->postgeocache(); 

    m_ok->reportKey("OKX4Test"); 

    return mgr.rc() ;
}
