#include "DetectorConstruction.hh"

// detector 
#include "G4VUserDetectorConstruction.hh"
#include "G4Element.hh"
#include "G4Material.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "G4SDManager.hh"

DetectorConstruction::DetectorConstruction( const char* sdname_ )
    :
    G4VUserDetectorConstruction(),
    sdname(strdup(sdname_))
{
}

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


G4Material* DetectorConstruction::MakeGlass()
{
    // from LXe example
    G4double a, z, density;
    G4int nelements;
    G4Element* H = new G4Element("H", "H", z=1., a=1.01*CLHEP::g/CLHEP::mole);
    G4Element* C = new G4Element("C", "C", z=6., a=12.01*CLHEP::g/CLHEP::mole);

    G4Material* mat = new G4Material("Glass", density=1.032*CLHEP::g/CLHEP::cm3,nelements=2);
    mat->AddElement(C,91.533*CLHEP::perCent);
    mat->AddElement(H,8.467*CLHEP::perCent);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    G4MaterialPropertyVector* ri = MakeGlassRI() ; 
    ri->SetSpline(false);
    mpt->AddProperty("RINDEX", ri);
    mat->SetMaterialPropertiesTable(mpt);

    return mat ; 
}

G4MaterialPropertyVector* DetectorConstruction::MakeGlassRI()
{
    using CLHEP::eV ;
 
    G4double photonEnergy[]   = { 2.034*eV , 2.885*eV, 4.136*eV };
    G4double refractiveIndex[] ={  1.49   , 1.49   , 1.49    };     

    assert(sizeof(photonEnergy) == sizeof(refractiveIndex));
    const G4int nEntries = sizeof(photonEnergy)/sizeof(G4double);

    return new G4MaterialPropertyVector(photonEnergy, refractiveIndex,nEntries ); 
}

G4MaterialPropertyVector* DetectorConstruction::MakeWaterRI()
{
    using CLHEP::eV ; 
    G4double photonEnergy[] =
                { 2.034*eV, 2.068*eV, 2.103*eV, 2.139*eV,
                  2.177*eV, 2.216*eV, 2.256*eV, 2.298*eV,
                  2.341*eV, 2.386*eV, 2.433*eV, 2.481*eV,
                  2.532*eV, 2.585*eV, 2.640*eV, 2.697*eV,
                  2.757*eV, 2.820*eV, 2.885*eV, 2.954*eV,
                  3.026*eV, 3.102*eV, 3.181*eV, 3.265*eV,
                  3.353*eV, 3.446*eV, 3.545*eV, 3.649*eV,
                  3.760*eV, 3.877*eV, 4.002*eV, 4.136*eV };

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
    using CLHEP::eV ; 
    G4double photonEnergy[] =
                { 2.034*eV, 2.068*eV, 2.103*eV, 2.139*eV,
                  2.177*eV, 2.216*eV, 2.256*eV, 2.298*eV,
                  2.341*eV, 2.386*eV, 2.433*eV, 2.481*eV,
                  2.532*eV, 2.585*eV, 2.640*eV, 2.697*eV,
                  2.757*eV, 2.820*eV, 2.885*eV, 2.954*eV,
                  3.026*eV, 3.102*eV, 3.181*eV, 3.265*eV,
                  3.353*eV, 3.446*eV, 3.545*eV, 3.649*eV,
                  3.760*eV, 3.877*eV, 4.002*eV, 4.136*eV };

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
    G4Box* so_1 = new G4Box("Obj",500.,500.,500.);
    G4LogicalVolume* lv_1 = new G4LogicalVolume(so_1,water,"Obj",0,0,0);
    G4VPhysicalVolume* pv_1 = new G4PVPlacement(0,G4ThreeVector(),lv_1 ,"Obj",lv_0,false,0);
    assert( pv_1 ); 

    G4Material* glass = MakeGlass();    // slab of sensitive glass in the water 
    G4Box* so_2 = new G4Box("Det",400.,400.,10.);  // half sizes 
    G4LogicalVolume* lv_2 = new G4LogicalVolume(so_2,glass,"Det",0,0,0);
    G4VPhysicalVolume* pv_2 = new G4PVPlacement(0,G4ThreeVector(0,0,100.),lv_2 ,"Det",lv_1,false,0);
    assert( pv_2 ); 

    G4SDManager* SDMan = G4SDManager::GetSDMpointerIfExist();        assert( SDMan && " SDMan should have been created before now " ); 
    G4VSensitiveDetector* sd = SDMan->FindSensitiveDetector(sdname); assert( sd && " failed for find sd with sdname " ); 
    lv_2->SetSensitiveDetector(sd); 

    return pv_0 ; 
}


