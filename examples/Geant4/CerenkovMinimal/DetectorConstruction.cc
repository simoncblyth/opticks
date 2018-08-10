#include "DetectorConstruction.hh"

// detector 
#include "G4VUserDetectorConstruction.hh"
#include "G4Element.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"


G4Material* DetectorConstruction::MakeWater()
{
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* water = new G4Material("Water", density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    water->AddElement(H, 2);
    water->AddElement(O, 1);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    G4MaterialPropertyVector* ri = MakeWaterRI() ; 
    ri->SetSpline(false);
    mpt->AddProperty("RINDEX", ri);
    water->SetMaterialPropertiesTable(mpt);
    return water ; 
}

G4Material* DetectorConstruction::MakeAir()
{
    G4double a, z, density;
    G4int nelements;
    G4Element* N = new G4Element("Nitrogen", "N", z=7 , a=14.01*CLHEP::g/CLHEP::mole);
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);

    G4Material* air = new G4Material("Air", density=1.29*CLHEP::mg/CLHEP::cm3, nelements=2);
    air->AddElement(N, 70.*CLHEP::perCent);
    air->AddElement(O, 30.*CLHEP::perCent);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    G4MaterialPropertyVector* ri = MakeAirRI() ; 
    ri->SetSpline(false);
    mpt->AddProperty("RINDEX", ri);
    air->SetMaterialPropertiesTable(mpt);
    return air ; 
}

G4MaterialPropertyVector* DetectorConstruction::MakeWaterRI()
{
    G4double photonEnergy[] =
                { 2.034*CLHEP::eV, 2.068*CLHEP::eV, 2.103*CLHEP::eV, 2.139*CLHEP::eV,
                  2.177*CLHEP::eV, 2.216*CLHEP::eV, 2.256*CLHEP::eV, 2.298*CLHEP::eV,
                  2.341*CLHEP::eV, 2.386*CLHEP::eV, 2.433*CLHEP::eV, 2.481*CLHEP::eV,
                  2.532*CLHEP::eV, 2.585*CLHEP::eV, 2.640*CLHEP::eV, 2.697*CLHEP::eV,
                  2.757*CLHEP::eV, 2.820*CLHEP::eV, 2.885*CLHEP::eV, 2.954*CLHEP::eV,
                  3.026*CLHEP::eV, 3.102*CLHEP::eV, 3.181*CLHEP::eV, 3.265*CLHEP::eV,
                  3.353*CLHEP::eV, 3.446*CLHEP::eV, 3.545*CLHEP::eV, 3.649*CLHEP::eV,
                  3.760*CLHEP::eV, 3.877*CLHEP::eV, 4.002*CLHEP::eV, 4.136*CLHEP::eV };

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
    return new G4MaterialPropertyVector(photonEnergy, refractiveIndex,nEntries ); 
}

G4MaterialPropertyVector* DetectorConstruction::MakeAirRI()
{
    G4double photonEnergy[] =
                { 2.034*CLHEP::eV, 2.068*CLHEP::eV, 2.103*CLHEP::eV, 2.139*CLHEP::eV,
                  2.177*CLHEP::eV, 2.216*CLHEP::eV, 2.256*CLHEP::eV, 2.298*CLHEP::eV,
                  2.341*CLHEP::eV, 2.386*CLHEP::eV, 2.433*CLHEP::eV, 2.481*CLHEP::eV,
                  2.532*CLHEP::eV, 2.585*CLHEP::eV, 2.640*CLHEP::eV, 2.697*CLHEP::eV,
                  2.757*CLHEP::eV, 2.820*CLHEP::eV, 2.885*CLHEP::eV, 2.954*CLHEP::eV,
                  3.026*CLHEP::eV, 3.102*CLHEP::eV, 3.181*CLHEP::eV, 3.265*CLHEP::eV,
                  3.353*CLHEP::eV, 3.446*CLHEP::eV, 3.545*CLHEP::eV, 3.649*CLHEP::eV,
                  3.760*CLHEP::eV, 3.877*CLHEP::eV, 4.002*CLHEP::eV, 4.136*CLHEP::eV };

    const G4int nEntries = sizeof(photonEnergy)/sizeof(G4double);

    G4double refractiveIndex[] =
            { 1.,  1.,  1.,  1., 
              1.,  1. , 1.,  1., 
              1.,  1. , 1.,  1., 
              1.,  1. , 1.,  1., 
              1.,  1.,  1.,  1., 
              1.,  1.,  1.,  1., 
              1.,  1.,  1.,  1., 
              1.,  1.,  1.,  1,};

    assert(sizeof(refractiveIndex) == sizeof(photonEnergy));
    return new G4MaterialPropertyVector(photonEnergy, refractiveIndex,nEntries ); 
}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    G4Material* air = MakeAir(); 
    G4Box* so_0 = new G4Box("World",1000.,1000.,1000.);
    G4LogicalVolume* lv_0 = new G4LogicalVolume(so_0,air,"World",0,0,0);
    G4VPhysicalVolume* pv_0 = new G4PVPlacement(0,G4ThreeVector(),lv_0 ,"World",0,false,0);

    G4Material* water = MakeWater(); 
    G4Box* so_1 = new G4Box("Obj",100.,100.,100.);
    G4LogicalVolume* lv_1 = new G4LogicalVolume(so_1,water,"Obj",0,0,0);
    G4VPhysicalVolume* pv_1 = new G4PVPlacement(0,G4ThreeVector(),lv_1 ,"Obj",lv_0,false,0);
    assert( pv_1 ); 

    return pv_0 ; 
}



