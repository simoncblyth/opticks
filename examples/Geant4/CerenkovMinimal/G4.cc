#include "G4.hh"
#include "Ctx.hh"

#include "G4RunManager.hh"
#include "DetectorConstruction.hh"
#include "L4Cerenkov.hh"
#include "PhysicsList.hh"
#include "PrimaryGeneratorAction.hh"
#include "SteppingAction.hh"
#include "EventAction.hh"
#include "TrackingAction.hh"


G4::G4()
    :
    ctx(new Ctx),
    rm(new G4RunManager),
    dc(new DetectorConstruction()),
    pl(new PhysicsList<L4Cerenkov>()),
    ga(NULL),
    ea(NULL),
    ta(NULL),
    sa(NULL)
{
    rm->SetUserInitialization(dc);
    rm->SetUserInitialization(pl);

    ga = new PrimaryGeneratorAction(ctx);
    ea = new EventAction(ctx) ;
    ta = new TrackingAction(ctx) ;
    sa = new SteppingAction(ctx) ;

    rm->SetUserAction(ga);
    rm->SetUserAction(ea);
    rm->SetUserAction(ta);
    rm->SetUserAction(sa);

    rm->Initialize(); 
}

void G4::beamOn(int nev)
{
    rm->BeamOn(nev); 
}


