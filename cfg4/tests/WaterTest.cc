#include "OPTICKS_LOG.hh"

#include "G4SystemOfUnits.hh"
#include "G4OpRayleigh.hh"
#include "G4Element.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4OpticalPhoton.hh"
#include "G4PhysicalConstants.hh"

/**
CWater
========

This is developed further in:

    X4MaterialWaterStandalone 
    X4MaterialWater


**/

struct CWater
{
    G4double                     density ;
    G4Material*                  Water ; 
    size_t                       WaterIndex ; 
    G4MaterialPropertiesTable*   WaterMPT ; 
    G4ParticleDefinition*        OpticalPhoton ; 
    G4OpRayleigh*                RayleighProcess ; 
    G4PhysicsTable*              thePhysicsTable; 
    G4PhysicsOrderedFreeVector*  rayleigh ; 
    std::array<double, 36>       fPP_Water_RIN ; 
    std::array<double, 36>       fWaterRINDEX ; 

    CWater(); 

    void initData(); 
    void dump() const ; 
    void rayleigh_scan() const ; 
    void rayleigh_scan2() const ; 

    G4double GetMeanFreePath( G4double energy ) const ; 
};

CWater::CWater()
    :
    density(1.000*g/cm3),
    Water(new G4Material("Water", density, 2)),
    WaterIndex(Water->GetIndex()),
    WaterMPT(new G4MaterialPropertiesTable()),
    OpticalPhoton(G4OpticalPhoton::Definition()),
    RayleighProcess(new G4OpRayleigh),
    thePhysicsTable(nullptr) 
{
    G4double z,a;
    G4Element* H  = new G4Element("Hydrogen" ,"H" , z= 1., a=   1.01*g/mole);
    G4Element* O  = new G4Element("Oxygen"   ,"O" , z= 8., a=  16.00*g/mole);

    Water->AddElement(H,2);
    Water->AddElement(O,1);
    Water->SetMaterialPropertiesTable(WaterMPT);  

    // jcv LSExpDetectorConstructionMaterial OpticalProperty
    initData(); 
    WaterMPT->AddProperty("RINDEX", fPP_Water_RIN.data(), fWaterRINDEX.data() ,36); 
    RayleighProcess->BuildPhysicsTable(*OpticalPhoton);  
    thePhysicsTable = RayleighProcess->GetPhysicsTable()  ; 
    rayleigh = static_cast<G4PhysicsOrderedFreeVector*>((*thePhysicsTable)(WaterIndex));
}


void CWater::dump() const 
{
    LOG(info) << " [ Water" ; 
    G4cout << *Water << G4endl ; 
    LOG(info) << " ] " ;

    LOG(info) << " [ G4OpRayleigh::DumpPhysicsTable " ; 
    RayleighProcess->DumpPhysicsTable(); 
    LOG(info) << " ] " ;
}

G4double CWater::GetMeanFreePath(G4double photonMomentum) const 
{
    return rayleigh ? rayleigh->Value( photonMomentum ) : DBL_MAX ; 
}


/**
CWater::rayleigh_scan
-----------------------

p104 of https://www.qmul.ac.uk/spa/media/pprc/research/Thesis_0.pdf
has measurements in the same ballpark 
 
**/

void CWater::rayleigh_scan() const 
{
    LOG(info) ; 
    for(unsigned w=200 ; w <= 800 ; w+=10 )
    { 
        G4double wavelength = double(w)*nm ; 
        G4double energy = h_Planck*c_light/wavelength ; 
        G4double length = GetMeanFreePath(energy); 

        std::cout 
            << "    " << std::setw(4) << w  << " nm "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << energy/eV << " eV "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << length/mm << " mm "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << length/m << " m  "
            << std::endl 
            ; 
    }
}

void CWater::rayleigh_scan2() const 
{
    LOG(info) ; 

    //G4double hc_eVnm = 1239.841875 ;
    G4double hc_eVnm = 1240. ;   // using 1240. regains 80nm and 800nm

    for(unsigned i=0 ; i < fPP_Water_RIN.size() ; i++ )
    { 
        G4double energy = fPP_Water_RIN[i] ; 
        G4double wavelength = h_Planck*c_light/energy ; 
        double w = wavelength/nm ; 
        double w2 =  hc_eVnm/(energy/eV) ;  

        G4double length = GetMeanFreePath(energy); 

        std::cout 
            << "    " << std::setw(10) << w  << " nm "
            << "    " << std::setw(10) << w2  << " nm "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << energy/eV << " eV "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << length/mm << " mm "
            << "    " << std::setw(10) << std::fixed << std::setprecision(3) << length/m << " m  "
            << std::endl 
            ; 
    }
}


void CWater::initData()   // jcv LSExpDetectorConstructionMaterial OpticalProperty
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






int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CWater water ; 
    water.dump(); 
    water.rayleigh_scan(); 
    water.rayleigh_scan2(); 

    return 0 ; 
}

// om-;TEST=WaterTest om-t 

