#include "DetectorConstruction.hh"

#include "G4RunManager.hh"


#include "G4NistManager.hh"

#include "G4MaterialTable.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4Sphere.hh"
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "globals.hh"
#include "G4UImanager.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4RotationMatrix.hh"

// /usr/local/env/g4/geant4.10.02.install/share/Geant4-10.2.0/examples/extended/optical/OpNovice/src/OpNoviceDetectorConstruction.cc

DetectorConstruction::~DetectorConstruction() {}


G4double DetectorConstruction::photonEnergy[] =
            { 2.034*eV, 2.068*eV, 2.103*eV, 2.139*eV,
              2.177*eV, 2.216*eV, 2.256*eV, 2.298*eV,
              2.341*eV, 2.386*eV, 2.433*eV, 2.481*eV,
              2.532*eV, 2.585*eV, 2.640*eV, 2.697*eV,
              2.757*eV, 2.820*eV, 2.885*eV, 2.954*eV,
              3.026*eV, 3.102*eV, 3.181*eV, 3.265*eV,
              3.353*eV, 3.446*eV, 3.545*eV, 3.649*eV,
              3.760*eV, 3.877*eV, 4.002*eV, 4.136*eV };

void DetectorConstruction::init()
{
    m_nistMan = G4NistManager::Instance(); 
   // m_nistMan->SetVerbose(5);

}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    m_vacuum = m_nistMan->FindOrBuildMaterial("G4_Galactic");
    m_vacuum->SetMaterialPropertiesTable(MakeVacuumProps());

    m_water  = m_nistMan->FindOrBuildMaterial("G4_WATER");
    m_water->SetMaterialPropertiesTable(MakeWaterProps());

    return ConstructDetector();
}

void DetectorConstruction::DumpDomain(const char* msg)
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

G4MaterialPropertiesTable* DetectorConstruction::MakeWaterProps()
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


G4MaterialPropertiesTable* DetectorConstruction::MakeVacuumProps()
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






G4VPhysicalVolume* DetectorConstruction::ConstructDetector()
{
  {
      G4double x = 1200.*mm;
      G4double y = 1200.*mm;
      G4double z = 1200.*mm;
      G4Box* solid = new G4Box("box_solid", x,y,z);

      m_box_log = new G4LogicalVolume(solid, m_vacuum,"box_log",0,0,0);
      m_box_phys = new G4PVPlacement(0,G4ThreeVector(), m_box_log, "box_phys",0,false,0);
  }

  {
     G4double radius = 100.*mm;
     G4Sphere* solid = new G4Sphere("sphere_solid", 0., radius, 0., twopi, 0., pi);    //size

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




