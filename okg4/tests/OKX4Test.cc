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

#include "OPTICKS_LOG.hh"

/**
OKX4Test : checking direct from G4 conversion, starting from a GDML loaded geometry
=======================================================================================

* TODO: update these notes, looks like now using pure G4 GDML parsing to init the geometry

The first Opticks is there just to work with CGDMLDetector
to load the GDML and apply fixups for missing material property tables
to provide the G4VPhysicalVolume world volume for checking 
the direct from G4.

It would be cleaner to do this with pure G4. Perhaps 
new Geant4 can avoid the fixups ?  Maybe not I think even
current G4 misses optical properties.

See :doc:`../../notes/issues/OKX4Test`

**/

/*
G4VPhysicalVolume* make_top(int argc, char** argv)
{

    Opticks* ok = new Opticks(argc, argv); 
    OpticksHub* hub = new OpticksHub(ok);
    OpticksQuery* query = ok->getQuery();
    CSensitiveDetector* sd = NULL ; 
    CGDMLDetector* detector  = new CGDMLDetector(hub, query, sd) ; 
    bool valid = detector->isValid();
    assert(valid);
    G4VPhysicalVolume* top = detector->Construct();
    return top ; 
}
*/


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    for(int i=0 ; i < argc ; i++)  
        LOG(info) << i << " " << argv[i] ; 


    const char* gdmlpath = PLOG::instance->get_arg_after("--gdmlpath", NULL) ; 
    if( gdmlpath == NULL )
    {
        LOG(fatal) << " --gdmlpath existing-path : is required " ; 
        return 0 ; 
    } 

    /**
    TODO: collect this gdmlpath into metadata stored in geocache, 
          to avoid having to extract it by parsing the argline 

    **/

    const char* csgskiplv = PLOG::instance->get_arg_after("--csgskiplv", NULL) ; 
    LOG(info) << " csgskiplv " << ( csgskiplv ? csgskiplv : "NONE" ) ;  


    const char* digestextra2 = PLOG::instance->get_arg_after("--digestextra", NULL) ; 
    LOG(info) << " digestextra2 " << ( digestextra2 ? digestextra2 : "NONE" ) ;  

    // need these prior to Opticks instanciation 


    LOG(info) << " parsing " << gdmlpath ; 
    G4VPhysicalVolume* top = CGDML::Parse( gdmlpath ) ; 
    assert(top);
    LOG(info) << "///////////////////////////////// " ; 


    const char* digestextra1 = csgskiplv ;    // kludge the digest to be sensitive to csgskiplv
    const char* spec = X4PhysicalVolume::Key(top, digestextra1, digestextra2 ) ; 

    Opticks::SetKey(spec);

    LOG(error) << " SetKey " << spec  ;   

    const char* argforce = "--tracer --nogeocache --xanalytic" ;
    // --nogeoache to prevent GGeo booting from cache 

    Opticks* ok = new Opticks(argc, argv, argforce);  // Opticks instanciation must be after Opticks::SetKey
    ok->configure();
    ok->enforceNoGeoCache(); 
 

    ok->profile("_OKX4Test:GGeo"); 

    GGeo* gg = new GGeo(ok) ;
    assert(gg->getMaterialLib());

    ok->profile("OKX4Test:GGeo"); 
    LOG(info) << " gg " << gg 
              << " gg.mlib " << gg->getMaterialLib()
              ;

    ok->profile("_OKX4Test:X4PhysicalVolume"); 

    X4PhysicalVolume xtop(gg, top) ;    // populates gg

    ok->profile("OKX4Test:X4PhysicalVolume"); 





    bool save_gltf = false ; 
    if(save_gltf)
    {
        int root = SSys::getenvint( "GLTF_ROOT", 3147 ); 
        const char* gltfpath = ok->getGLTFPath(); 
        ok->profile("_OKX4Test:GGeoGLTF"); 
        GGeoGLTF::Save(gg, gltfpath, root ); 
        ok->profile("OKX4Test:GGeoGLTF"); 
    }


    gg->prepare();   // merging meshes, closing libs

   // not OKG4Mgr as no need for CG4 

    ok->profile("_OKX4Test:OKMgr"); 
    OKMgr mgr(argc, argv);  // OpticksHub inside here picks up the gg (last GGeo instanciated) via GGeo::GetInstance 
    ok->profile("OKX4Test:OKMgr"); 
    //mgr.propagate();
    mgr.visualize();   

    assert( GGeo::GetInstance() == gg );
    gg->reportMeshUsage();
    gg->save();

    Opticks* oki = Opticks::GetInstance() ; 
    ok->saveProfile();
    ok->postgeocache(); 

    assert( oki == ok ) ; 
    ok->reportGeoCacheCoordinates(); 


    return mgr.rc() ;

}
