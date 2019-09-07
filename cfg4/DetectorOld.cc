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

#include "CFG4_BODY.hh"





G4double Detector::photonEnergy[] =
            { 2.034*eV, 2.068*eV, 2.103*eV, 2.139*eV,
              2.177*eV, 2.216*eV, 2.256*eV, 2.298*eV,
              2.341*eV, 2.386*eV, 2.433*eV, 2.481*eV,
              2.532*eV, 2.585*eV, 2.640*eV, 2.697*eV,
              2.757*eV, 2.820*eV, 2.885*eV, 2.954*eV,
              3.026*eV, 3.102*eV, 3.181*eV, 3.265*eV,
              3.353*eV, 3.446*eV, 3.545*eV, 3.649*eV,
              3.760*eV, 3.877*eV, 4.002*eV, 4.136*eV };








void Detector::DumpDomain(const char* msg)
{
  const G4int nEntries = sizeof(photonEnergy)/sizeof(G4double);
  G4cout << msg << " " << nEntries << G4endl ;  

  for(unsigned int i=0 ; i < nEntries ; i++)
  { 
      G4double wavelength = h_Planck*c_light/photonEnergy[i];
      G4cout << std::setw(4) << i 
             << " eV " << photonEnergy[i]/eV 
             << " en " << photonEnergy[i]
             << " eV " << eV  
             << " nm " << wavelength/nm  
             << " wl " << wavelength 
             << " nm " << nm
             << G4endl 
             ;
  } 
}





G4MaterialPropertiesTable* Detector::MakeWaterProps()
{
    const G4int nEntries = sizeof(photonEnergy)/sizeof(G4double);

    G4double refractiveIndex[] =
            { 1.3435, 1.344,  1.3445, 1.345,  1.3455,
              1.346,  1.3465, 1.347,  1.3475, 1.348,
              1.3485, 1.3492, 1.35,   1.3505, 1.351,
              1.3518, 1.3522, 1.3530, 1.3535, 1.354,
              1.3545, 1.355,  1.3555, 1.356,  1.3568,
              1.3572, 1.358,  1.3585, 1.359,  1.3595,
              1.36,   1.3608};

    assert(sizeof(refractiveIndex) == sizeof(photonEnergy));

    G4double absorption[] =
           {3.448*m,  4.082*m,  6.329*m,  9.174*m, 12.346*m, 13.889*m,
           15.152*m, 17.241*m, 18.868*m, 20.000*m, 26.316*m, 35.714*m,
           45.455*m, 47.619*m, 52.632*m, 52.632*m, 55.556*m, 52.632*m,
           52.632*m, 47.619*m, 45.455*m, 41.667*m, 37.037*m, 33.333*m,
           30.000*m, 28.500*m, 27.000*m, 24.500*m, 22.000*m, 19.500*m,
           17.500*m, 14.500*m };

    assert(sizeof(absorption) == sizeof(photonEnergy));

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX",    photonEnergy, refractiveIndex, nEntries)->SetSpline(true); 
    mpt->AddProperty("ABSLENGTH", photonEnergy, absorption,      nEntries)->SetSpline(true);
    //mpt->DumpTable();

    return mpt ;  
}


G4MaterialPropertiesTable* Detector::MakeVacuumProps()
{
    const G4int nEntries = sizeof(photonEnergy)/sizeof(G4double);

    G4double* refractiveIndex = new G4double[nEntries];
    G4double* absorption = new G4double[nEntries];
    for(unsigned int i=0 ; i<nEntries ; i++) 
    {
        refractiveIndex[i] = 1.0 ;
        absorption[i] = 10000.0*m ;
    }

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX",    photonEnergy, refractiveIndex, nEntries)->SetSpline(true); 
    mpt->AddProperty("ABSLENGTH", photonEnergy, absorption,      nEntries)->SetSpline(true);

    delete [] refractiveIndex  ;
    delete [] absorption ;

    //mpt->DumpTable();
    return mpt ;  
}





G4VPhysicalVolume* Detector::ConstructDetector_Old()
{
  float extent = m_center_extent.w ;

  LOG(info) << "Detector::ConstructDetector extent " << extent << " mm " << mm ;  
  {
  
      G4double x = extent*mm;
      G4double y = extent*mm;
      G4double z = extent*mm;
      G4Box* solid = new G4Box("box_solid", x,y,z);

      m_box_log = new G4LogicalVolume(solid, m_vacuum,"box_log",0,0,0);
      m_box_phys = new G4PVPlacement(0,G4ThreeVector(), m_box_log, "box_phys",0,false,0);
  }

  {
     G4double radius = 100.*mm;
     G4Sphere* solid = new G4Sphere("sphere_solid", 0., radius, 0., twopi, 0., pi);   

     G4LogicalVolume* log = new G4LogicalVolume(solid, m_water, "sphere_log");
    
     m_sphere_phys = new G4PVPlacement(
                             0,                         //no rotation
                             G4ThreeVector(),           //at (0,0,0)
                             log,                       //logical volume
                             "sphere_phys",             //name
                             m_box_log,                 //mother  volume
                             false,                     //no boolean operation
                             0);                        //copy number
  } 
  return m_box_phys ;
}


G4VPhysicalVolume* Detector::Construct_Old()
{
    m_vacuum = m_nistMan->FindOrBuildMaterial("G4_Galactic");
    m_vacuum->SetMaterialPropertiesTable(MakeVacuumProps());

    m_water  = m_nistMan->FindOrBuildMaterial("G4_WATER");
    m_water->SetMaterialPropertiesTable(MakeWaterProps());

    return ConstructDetector();
}

