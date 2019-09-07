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

// TEST=G4PhysicalConstantsTest om-t 

#include "G4Types.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    G4double kineticEnergy_eV = 1.0 ; 
    G4double kineticEnergy = kineticEnergy_eV*eV ; 

    G4double hc_eVnm = 1239.841875 ;  
    G4double wavelength_nm_0 = hc_eVnm/kineticEnergy_eV ;  // simplest way 
    // suspect the above is more robust when using floats 

    G4double wavelength_nm_1 = h_Planck*c_light/kineticEnergy/nm ;  // standard G4 way : without any regard for numerical robustness

    LOG(info) 
         << " kineticEnergy " 
         << std::fixed << kineticEnergy
         << " kineticEnergy/eV " 
         << std::fixed << kineticEnergy/eV
         << " kineticEnergy/MeV " 
         << std::fixed << kineticEnergy/MeV
         << " wavelength_nm_1 " 
         << std::fixed << wavelength_nm_1
         << " wavelength_nm_0 " 
         << std::fixed << wavelength_nm_0
         ;

    return 0 ; 
}

