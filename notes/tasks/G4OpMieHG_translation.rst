G4OpMieHG
==========

Overview
-----------

As MieHG scattering is not important for many experiements, all 
changes to support it will need to be made in an optional 
manner for example with compilation options to include it.


Steps to support MieHG scattering 
-----------------------------------

Geant4 code G4OpMieHG looks like it would be straightforward to port it to CUDA within Opticks.
What needs to be done:


1. modify Opticks material property handling and boundary texture to include the 
   additional kMIEHG properties  
   Changes needed in ggeo/GMaterialLib ggeo/GBndLib extg4/X4MaterialLib ...

   The boundary texture currently has two float4 with five of the eight properties occupies, 
   the four properties needed for MIE scattering would require changing the boundary texture
   shape to accomodate these.  

2. implement the CUDA optixrap/cu/mie.h based on source/processes/optical/src/G4OpMieHG.cc
   in an analogous manner to how optixrap/cu/rayleigh.h which is based on source/processes/optical/src/G4OpRayleigh.cc 

3. modify oxrap/cu/generate.cu to access the expanded boundary texture and add the process by 
   adding more random generation to yield a miescattering distance and compare with aborption and rayleight scattering 
   lengths.

4. validate the ported code by comparisons with Geant4 


Geant4 sources relevant 
-------------------------

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



  
