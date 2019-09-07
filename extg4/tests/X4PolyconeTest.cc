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

#include "OPTICKS_LOG.hh"

#include "Opticks.hh"
#include "G4Polycone.hh"
#include "X4Solid.hh"
#include "X4.hh"
#include "NNode.hpp"



G4VSolid* make_Polycone()
{
    G4double phiStart = 0 ; 
    G4double phiTotal = CLHEP::twopi ; 
    G4int numZPlanes = 4 ; 

    double zPlane[] = { 3937, 4000.02, 4000.02, 4094.62 } ;
    double rInner[] = {    0,       0,       0,     0   } ;
    double rOuter[] = { 2040,    2040,    1930,   125   } ;


    G4Polycone* pc = new G4Polycone("poly", phiStart, phiTotal, numZPlanes, zPlane, rInner, rOuter );

    G4VSolid* so = pc ; 

    return so ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv); 

    G4VSolid* so = make_Polycone() ; 

    std::cout << *so << std::endl ; 

    X4Solid xs(so, &ok, true) ; 

    nnode* root = xs.root(); 

    root->dump_g4code();  

    return 0 ; 
}

