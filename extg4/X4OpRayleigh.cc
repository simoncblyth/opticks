#include "PLOG.hh"

#include "G4Material.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpRayleigh.hh"

#include "X4OpRayleigh.hh"

/**
X4OpRayleigh::WaterScatteringLength
------------------------------------

A fallback calulation is used only when the G4Material named "Water"
does not have a "RAYLEIGH" property.

**/

G4PhysicsVector* X4OpRayleigh::WaterScatteringLength() // static 
{
    X4OpRayleigh proc ; 
    return proc.rayleigh ; 
}  

/**
X4OpRayleigh::GetFromPhysicsTable
----------------------------------

G4PhysicsTable ISA std::vector<G4PhysicsVector*>

Formerly returned G4PhysicsOrderedFreeVector.

See notes/issues/X4MaterialWater_g4_11_compilation_error_G4PhysicsFreeVector_G4PhysicsOrderedFreeVector.rst

**/

G4PhysicsVector* X4OpRayleigh::GetFromPhysicsTable(const G4OpRayleigh* proc, size_t index ) // static 
{
    G4PhysicsTable* thePhysicsTable = proc->GetPhysicsTable()  ; 
    G4PhysicsVector* vec = (*thePhysicsTable)(index) ; 
    return vec ;
}

X4OpRayleigh::X4OpRayleigh()
    :
    Water(G4Material::GetMaterial("Water")),
    WaterIndex(Water ? Water->GetIndex() : 0), 
    OpticalPhoton(G4OpticalPhoton::Definition()), 
    RayleighProcess(new G4OpRayleigh), 
    rayleigh(nullptr)
{
    init(); 
}

void X4OpRayleigh::init()
{
    assert(Water); 
    RayleighProcess->BuildPhysicsTable(*OpticalPhoton);  
    rayleigh = GetFromPhysicsTable(RayleighProcess, WaterIndex) ; 
}

void X4OpRayleigh::dump() const 
{
    LOG(info) << " [ WaterIndex " << WaterIndex ;
    G4cout << *Water << G4endl ; 
    LOG(info) << " ] " ;

    LOG(info) << " [ DumpPhysicsTable " ; 
    RayleighProcess->DumpPhysicsTable(); 
    LOG(info) << " ] " ;
}

