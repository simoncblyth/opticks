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

// TEST=CGenstepSourceTest om-t

#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "NPY.hpp"

#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4ThreeVector.hh"


#include "CGenstepSource.hh"

#include "Opticks.hh"
#include "OpticksMode.hh"
#include "OpticksHub.hh"
#include "CMaterialLib.hh"



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

/*
    const char* def = "/usr/local/opticks/opticksdata/gensteps/dayabay/natural/1.npy" ; 
    const char* path = argc > 1 ? argv[1] : def ; 

    NPY<float>* np = NPY<float>::load(path) ; 
    if(np == NULL) return 0 ; 
*/  
  
    Opticks ok(argc, argv);
    ok.setModeOverride( OpticksMode::CFG4_MODE );  // override COMPUTE/INTEROP mode, as those do not apply to CFG4 
    OpticksHub hub(&ok) ; 
    CMaterialLib* clib = new CMaterialLib(&hub);
    clib->convert();
    // TODO: a more direct way to get to a refractive index, than the above that loads the entire geometry  

    LOG(error) << "--------------------------------" ; 


    if(!ok.hasKey())
    {
        LOG(fatal) << " currently this test is for keyed (direct mode) running only " ; 
        return 0 ; 
    }


    const char* gsp = ok.getDirectGenstepPath() ; 
    LOG(error) << " gsp " << gsp ; 

    NPY<float>* np = ok.loadDirectGenstep() ; 
    if(np == NULL) return 0 ; 


    CGenstepSource* gsrc = new CGenstepSource( &ok, np ) ; 

    G4Event* event = new G4Event ; 

    gsrc->GeneratePrimaryVertex(event) ;  

    G4int num_prim = event->GetNumberOfPrimaryVertex() ; 
    assert( num_prim > 0 ) ; 

    LOG(info) << " num_prim " << num_prim  ; 

    for(G4int i=0 ; i < num_prim ; i++)
    {
        G4PrimaryVertex* vertex = event->GetPrimaryVertex(i) ; 
        //vertex->Print(); 
        const G4ThreeVector& position = vertex->GetPosition() ;

        G4int num_part = vertex->GetNumberOfParticle() ; 
        assert( num_part == 1 );

        G4PrimaryParticle* particle = vertex->GetPrimary();  

        G4double kineticEnergy = particle->GetKineticEnergy() ;  
        G4double wavelength = h_Planck*c_light/kineticEnergy ; 
 
        std::cout 
            << " wavelength " << wavelength/nm 
            << " position " << position
            << std::endl 
            ;

    }
  


    return 0 ; 
}



