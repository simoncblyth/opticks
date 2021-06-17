
#include <iostream>
#include <iomanip>

#include "PLOG.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4Material.hh"

#include "X4OpRayleigh.hh"
#include "X4PhysicsOrderedFreeVector.hh"
#include "X4MaterialWater.hh"

const plog::Severity X4MaterialWater::LEVEL = PLOG::EnvLevel("X4MaterialWater", "DEBUG" ); 

/**
X4MaterialWater::IsApplicable
------------------------------

**/

bool X4MaterialWater::IsApplicable()  // static 
{
    G4PhysicsOrderedFreeVector* RAYLEIGH = GetRAYLEIGH();
    G4PhysicsOrderedFreeVector* RINDEX = GetRINDEX();
    bool applicable = RAYLEIGH == nullptr && RINDEX != nullptr ; 

    LOG(LEVEL)
        << " RAYLEIGH " << RAYLEIGH 
        << " RINDEX " << RINDEX 
        << " applicable " << applicable 
        ;

    return applicable ; 
}

G4PhysicsOrderedFreeVector* X4MaterialWater::GetRAYLEIGH(){ return GetProperty(kRAYLEIGH) ; }
G4PhysicsOrderedFreeVector* X4MaterialWater::GetRINDEX(){  return GetProperty(kRINDEX) ; }

G4PhysicsOrderedFreeVector* X4MaterialWater::GetProperty(const G4int index)
{
    G4Material* Water = G4Material::GetMaterial("Water");
    if(Water == nullptr) return nullptr ; 
    G4MaterialPropertiesTable* WaterMPT = Water->GetMaterialPropertiesTable() ; 
    if(WaterMPT == nullptr) return nullptr ; 
    G4PhysicsOrderedFreeVector* PROP = WaterMPT->GetProperty(index) ;
    return PROP ;     
}

X4MaterialWater::X4MaterialWater()
    :
    Water(G4Material::GetMaterial("Water")),
    WaterMPT(Water ? Water->GetMaterialPropertiesTable() : nullptr),
    rayleigh0(WaterMPT ? static_cast<G4PhysicsOrderedFreeVector*>(WaterMPT->GetProperty(kRAYLEIGH)) : nullptr ),
    rayleigh(rayleigh0 ? rayleigh0 : X4OpRayleigh::WaterScatteringLength() )
{
    init(); 
}

/**
X4MaterialWater::init
----------------------

When the G4Material named "Water" does not have a RAYLEIGH property vector this adds the one calculated 
from the RINDEX by G4OpRayleigh and fished out of the process physics table.

Note that GetMaterialPropertiesTable::AddProperty just inserts the reference into the property map
of the material with no content copying. So any subsequent changes to the vector will change the 
properties of the material.

**/
void X4MaterialWater::init()
{
    if( rayleigh0 == nullptr ) 
    {
        LOG(LEVEL) << " adding RAYLEIGH property to \"Water\" G4Material " ; 
        WaterMPT->AddProperty("RAYLEIGH", static_cast<G4MaterialPropertyVector*>(rayleigh) ); 
    }
    else
    {
        LOG(LEVEL) << " nothing to do as the \"Water\" G4Material already has RAYLEIGH property " ; 
    }
}

void X4MaterialWater::dump() const 
{
    LOG(info) << " [ Water " ; 
    G4cout << *Water << G4endl ; 
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

G4PhysicsOrderedFreeVector isa G4PhysicsVector

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

void X4MaterialWater::rayleigh_scan2() const
{
    X4PhysicsOrderedFreeVector rayleighx(rayleigh); 
    LOG(info) << rayleighx.desc() ; 
} 
void X4MaterialWater::changeRayleighToMidBin()
{
    X4PhysicsOrderedFreeVector rayleighx(rayleigh); 
    rayleighx.changeAllToMidBinValue() ;     
}


