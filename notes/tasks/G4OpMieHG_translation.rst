G4OpMieHG
==========


::

    epsilon:geant4.10.04.p02 blyth$ find source -name '*.cc' -exec grep -H kMIEHG {} \;
    source/processes/optical/src/G4OpMieHG.cc:              aMaterialPropertyTable->GetConstProperty(kMIEHG_FORWARD);
    source/processes/optical/src/G4OpMieHG.cc:              aMaterialPropertyTable->GetConstProperty(kMIEHG_BACKWARD);
    source/processes/optical/src/G4OpMieHG.cc:              aMaterialPropertyTable->GetConstProperty(kMIEHG_FORWARD_RATIO);
    epsilon:geant4.10.04.p02 blyth$ find source -name '*.hh' -exec grep -H kMIEHG {} \;
    source/materials/include/G4MaterialPropertiesIndex.hh:  kMIEHG,                      // Mie scattering length
    source/materials/include/G4MaterialPropertiesIndex.hh:  kMIEHG_FORWARD,               // forward angle of Mie scattering based on Henyey-Greenstein phase function
    source/materials/include/G4MaterialPropertiesIndex.hh:  kMIEHG_BACKWARD,              // backward angle of Mie scattering based on Henyey-Greenstein phase function
    source/materials/include/G4MaterialPropertiesIndex.hh:  kMIEHG_FORWARD_RATIO,	        // ratio of the MIEHG forward scattering 
    epsilon:geant4.10.04.p02 blyth$ 


    epsilon:geant4.10.04.p02 blyth$ find source -name '*.cc' -exec grep -H  G4MaterialPropertiesIndex {} \+
    source/materials/src/G4MaterialPropertiesTable.cc:  // the corresponding enums in G4MaterialPropertiesIndex.hh
    epsilon:geant4.10.04.p02 blyth$ 


