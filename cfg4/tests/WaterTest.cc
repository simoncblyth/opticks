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

