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

