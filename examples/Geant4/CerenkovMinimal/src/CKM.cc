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

#include "CKM.hh"

#include "SensitiveDetector.hh"
#include "DetectorConstruction.hh"
#include "L4Cerenkov.hh"
#include "CKMScintillation.h"
#include "PhysicsList.hh"
#include "PrimaryGeneratorAction.hh"


#include "G4Opticks.hh"
#include "G4OpticksRecorder.hh"
#include "PLOG.hh"
#include "SSys.hh"

#include "CManager.hh"

#include "RAction.hh"
#include "EAction.hh"
#include "TAction.hh"
#include "SAction.hh"


CKM::CKM()
    :
    g4ok(new G4Opticks),
    okr(new G4OpticksRecorder),
    rm(new G4RunManager),
    sdn("SD0"),
    sd(new SensitiveDetector(sdn)),
    dc(new DetectorConstruction(sdn)),
    pl(new PhysicsList<L4Cerenkov,CKMScintillation>()),
    ga(nullptr),
    ra(nullptr),
    ea(nullptr),
    ta(nullptr),
    sa(nullptr)
{
    init();
}

void CKM::init()
{
    rm->SetUserInitialization(dc);
    rm->SetUserInitialization(pl);

    ga = new PrimaryGeneratorAction();
    ra = new RAction(okr) ; 
    ea = new EAction(okr) ; 
    ta = new TAction(okr) ; 
    sa = new SAction(okr) ; 

    rm->SetUserAction(ga);
    rm->SetUserAction(ra);
    rm->SetUserAction(ea);
    rm->SetUserAction(ta);
    rm->SetUserAction(sa);

    rm->Initialize(); 

    const G4VPhysicalVolume* world = dc->getWorld(); 
    assert(world);  
    setup_G4Opticks(world); 
}


CKM::~CKM()
{
    G4GeometryManager::GetInstance()->OpenGeometry(); 
}

void CKM::beamOn(int nev)
{
    rm->BeamOn(nev); 
}



void CKM::setup_G4Opticks(const G4VPhysicalVolume* world )
{
    assert( world ) ; 

    bool standardize_geant4_materials = false ;   
    const char* embedded_commandline_extra = SSys::getenvvar("CKM_OPTICKS_EXTRA", "" );   

    LOG(info) << "embedded_commandline_extra (CKM_OPTICKS_EXTRA)" << embedded_commandline_extra ; 

    g4ok->setEmbeddedCommandLineExtra(embedded_commandline_extra);
    g4ok->setGeometry(world, standardize_geant4_materials );    

    const std::vector<G4PVPlacement*>& sensor_placements = g4ok->getSensorPlacements() ;
    for(unsigned i=0 ; i < sensor_placements.size()  ; i++)
    {
        float efficiency_1 = 0.5f ; 
        float efficiency_2 = 1.0f ; 
        int sensor_cat = -1 ;                   // -1:means no angular efficiency info 
        int sensor_identifier = 0xc0ffee + i ;  // mockup a detector specific identifier
        unsigned sensorIndex = 1+i ;            // 1-based
        g4ok->setSensorData( sensorIndex, efficiency_1, efficiency_2, sensor_cat, sensor_identifier );  
    }
}



