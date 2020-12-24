/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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


