#include "G4SystemOfUnits.hh"
#include "G4Element.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "X4MaterialWaterStandalone.hh"
#include "PLOG.hh"

X4MaterialWaterStandalone::X4MaterialWaterStandalone()
    :
    density(1.000*g/cm3),
    Water(new G4Material("Water", density, 2)),
    WaterMPT(new G4MaterialPropertiesTable())
{
    init(); 
}

void X4MaterialWaterStandalone::init()
{
    G4double z,a;
    G4Element* H  = new G4Element("Hydrogen" ,"H" , z= 1., a=   1.01*g/mole);
    G4Element* O  = new G4Element("Oxygen"   ,"O" , z= 8., a=  16.00*g/mole);

    Water->AddElement(H,2);
    Water->AddElement(O,1);
    Water->SetMaterialPropertiesTable(WaterMPT);  

    initData();
    WaterMPT->AddProperty("RINDEX", fPP_Water_RIN.data(), fWaterRINDEX.data() ,36); 
}

void X4MaterialWaterStandalone::dump() const 
{
    LOG(info) << " [ Water" ; 
    G4cout << *Water << G4endl ; 
    LOG(info) << " ] " ;
}


void X4MaterialWaterStandalone::initData()   // jcv LSExpDetectorConstructionMaterial OpticalProperty
{
    fPP_Water_RIN = {{
      1.55*eV,
      2.034*eV,
      2.068*eV,
      2.103*eV,
      2.139*eV,
      2.177*eV,
      2.216*eV,
      2.256*eV,
      2.298*eV,
      2.341*eV,
      2.386*eV,
      2.433*eV,
      2.481*eV,
      2.532*eV,
      2.585*eV,
      2.640*eV,
      2.697*eV,
      2.757*eV,
      2.820*eV,
      2.885*eV,
      2.954*eV,
      3.026*eV,
      3.102*eV,
      3.181*eV,
      3.265*eV,
      3.353*eV,
      3.446*eV,
      3.545*eV,
      3.649*eV,
      3.760*eV,
      3.877*eV,
      4.002*eV,
      4.136*eV,
      6.20*eV,
      10.33*eV,
      15.50*eV
  }};

  fWaterRINDEX = {{
      1.3333 ,
      1.3435 ,
      1.344  ,
      1.3445 ,
      1.345  ,
      1.3455 ,
      1.346  ,
      1.3465 ,
      1.347  ,
      1.3475 ,
      1.348  ,
      1.3485 ,
      1.3492 ,
      1.35   ,
      1.3505 ,
      1.351  ,
      1.3518 ,
      1.3522 ,
      1.3530 ,
      1.3535 ,
      1.354  ,
      1.3545 ,
      1.355  ,
      1.3555 ,
      1.356  ,
      1.3568 ,
      1.3572 ,
      1.358  ,
      1.3585 ,
      1.359  ,
      1.3595 ,
      1.36   ,
      1.3608 ,
      1.39,
      1.33,
      1.33
  }};
}

