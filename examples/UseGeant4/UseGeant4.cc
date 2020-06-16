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
#include "G4Version.hh"
#include "G4Polycone.hh"

using CLHEP::deg ; 


void dump_version()
{
    std::cout << "G4VERSION_NUMBER " << G4VERSION_NUMBER << std::endl ; 
    std::cout << "G4VERSION_TAG    " << G4VERSION_TAG << std::endl ; 
    std::cout << "G4Version        " << G4Version << std::endl ; 
    std::cout << "G4Date           " << G4Date << std::endl ; 
}


void make_polycone_0()
{
     G4double phiStart = 0.00*deg ; 
     G4double phiTotal = 360.00*deg ; 
     G4int numRZ = 2 ; 
     G4double r[] = {50.000999999999998, 75.82777395122217} ; 
     G4double z[] = {-19.710672039327765, 19.710672039327765} ; 
    
     G4Polycone* pc = new G4Polycone("name", phiStart, phiTotal, numRZ, r, z ); 
     G4cout << *pc << std::endl ; 
}

void make_polycone_1()
{
     G4double phiStart = 0.00*deg ; 
     G4double phiTotal = 360.00*deg ; 
     G4int numZPlanes = 2 ; 

     G4double zPlane[] = {-19.710672039327765, 19.710672039327765} ; 
     G4double rInner[] = {0.0, 0.0} ; 
     G4double rOuter[] = {50.000999999999998, 75.82777395122217} ; 
    
     G4Polycone* pc = new G4Polycone("name", phiStart, phiTotal, numZPlanes, zPlane, rInner, rOuter ); 
     G4cout << *pc << std::endl ; 

}





int main()
{
    dump_version(); 
    //make_polycone_0(); 
    make_polycone_1(); 

    return 0 ; 
}
