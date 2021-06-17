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

G4PhysicsOrderedFreeVector* X4OpRayleigh::WaterScatteringLength() // static 
{
    X4OpRayleigh proc ; 
    return proc.rayleigh ; 
}  

G4PhysicsOrderedFreeVector* X4OpRayleigh::GetFromPhysicsTable(const G4OpRayleigh* proc, size_t index ) // static 
{
    G4PhysicsTable* thePhysicsTable = proc->GetPhysicsTable()  ; 
    return static_cast<G4PhysicsOrderedFreeVector*>((*thePhysicsTable)(index));
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

