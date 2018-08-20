#include "G4RunManager.hh"
#include "G4GeometryManager.hh"

#include "G4.hh"
#include "Ctx.hh"

#include "SensitiveDetector.hh"
#include "DetectorConstruction.hh"
#include "L4Cerenkov.hh"
#include "PhysicsList.hh"
#include "PrimaryGeneratorAction.hh"

#include "RunAction.hh"
#include "EventAction.hh"
#include "TrackingAction.hh"
#include "SteppingAction.hh"

G4::G4(int nev)
    :
    ctx(new Ctx),
    rm(new G4RunManager),
    sdn("SD0"),
    sd(new SensitiveDetector(sdn)),
    dc(new DetectorConstruction(sdn)),
    pl(new PhysicsList<L4Cerenkov>()),
    ga(NULL),
    ra(NULL),
    ea(NULL),
    ta(NULL),
    sa(NULL)
{
    rm->SetUserInitialization(dc);
    rm->SetUserInitialization(pl);

    ga = new PrimaryGeneratorAction(ctx);
    ra = new RunAction(ctx) ;
    ea = new EventAction(ctx) ;
    ta = new TrackingAction(ctx) ;
    sa = new SteppingAction(ctx) ;

    rm->SetUserAction(ga);
    rm->SetUserAction(ra);
    rm->SetUserAction(ea);
    rm->SetUserAction(ta);
    rm->SetUserAction(sa);

    rm->Initialize(); 

    beamOn(nev); 
}


G4::~G4()
{
    G4GeometryManager::GetInstance()->OpenGeometry(); 
}


void G4::beamOn(int nev)
{
    rm->BeamOn(nev); 
}


