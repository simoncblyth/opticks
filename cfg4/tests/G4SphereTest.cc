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

#include <iostream>

#include "G4String.hh"
#include "G4Sphere.hh"
#include "G4Polyhedron.hh"
#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    G4String name("sphere");

    G4double pRmin = 0. ; 
    G4double pRmax = 100. ; 
    G4double pSPhi = 0. ; 
    G4double pDPhi = 2.*CLHEP::pi ;

    G4double pSTheta = 0.f ; 
    G4double pDTheta = CLHEP::pi ;


    G4Sphere sp(name, pRmin, pRmax, pSPhi, pDPhi, pSTheta, pDTheta );

    std::cout << sp << std::endl ; 

    
    G4Polyhedron* poly = sp.CreatePolyhedron() ;

    G4cout << *poly << G4endl ;  




    return 0 ; 
}
