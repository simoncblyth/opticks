
#include <iostream>
#include <iomanip>

#include "PLOG.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpRayleigh.hh"

#include "X4MaterialWater.hh"

X4MaterialWater::X4MaterialWater()
    :
    Water(G4Material::GetMaterial("Water")),
    WaterIndex(Water ? Water->GetIndex() : 0),
    OpticalPhoton(G4OpticalPhoton::Definition()),
    RayleighProcess(new G4OpRayleigh),
    thePhysicsTable(nullptr) 
{
    init();
}

void X4MaterialWater::init()
{
    assert(Water); 
    RayleighProcess->BuildPhysicsTable(*OpticalPhoton);  
    thePhysicsTable = RayleighProcess->GetPhysicsTable()  ; 
    rayleigh = static_cast<G4PhysicsOrderedFreeVector*>((*thePhysicsTable)(WaterIndex));
}

void X4MaterialWater::dump() const 
{
    LOG(info) << " [ WaterIndex " << WaterIndex  ; 
    G4cout << *Water << G4endl ; 
    LOG(info) << " ] " ;

    LOG(info) << " [ G4OpRayleigh::DumpPhysicsTable " ; 
    RayleighProcess->DumpPhysicsTable(); 
    LOG(info) << " ] " ;
}

G4double X4MaterialWater::GetMeanFreePath(G4double photonMomentum) const 
{
    return rayleigh ? rayleigh->Value( photonMomentum ) : DBL_MAX ; 
}

/**
X4MaterialWater::rayleigh_scan
--------------------------------

p104 of https://www.qmul.ac.uk/spa/media/pprc/research/Thesis_0.pdf
has measurements in the same ballpark 
 
**/

void X4MaterialWater::rayleigh_scan() const 
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

 
