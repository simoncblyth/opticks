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


#include "Ctx.hh"
#include "OpHit.hh"
#include "SensitiveDetector.hh"

#include "G4OpticalPhoton.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4SDManager.hh"


#ifdef WITH_OPTICKS
#include "G4Opticks.hh"
#endif


const char* SensitiveDetector::SDName = NULL ; 
const char* SensitiveDetector::collectionNameA = "OpHitCollectionA" ;
const char* SensitiveDetector::collectionNameB = "OpHitCollectionB" ;

SensitiveDetector::SensitiveDetector(const char* name) 
    :
    G4VSensitiveDetector(name),
    m_hit_count(0)
{
    SDName = strdup(name) ; 
    collectionName.insert(collectionNameA); 
    collectionName.insert(collectionNameB); 

    G4SDManager* SDMan = G4SDManager::GetSDMpointer() ;
    SDMan->AddNewDetector(this); 
}

/**
SensitiveDetector::ProcessHits
-------------------------------

Collects Geant4 hits into Opticks collection.

**/

G4bool SensitiveDetector::ProcessHits(G4Step* step,G4TouchableHistory* )
{
    G4Track* track = step->GetTrack();
    if (track->GetDefinition() != G4OpticalPhoton::Definition()) return false ; 

    G4double ene = step->GetTotalEnergyDeposit();
    G4StepPoint* point = step->GetPreStepPoint();
    G4double time = point->GetGlobalTime(); 
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    m_hit_count += 1 ; 

#ifdef WITH_OPTICKS
    {
        G4double energy = point->GetKineticEnergy();
        G4double wavelength = h_Planck*c_light/energy ;
        G4double weight = 1.0 ;
        G4int flags_x = 0 ; 
        G4int flags_y = 0 ; 
        G4int flags_z = 0 ; 
        G4int flags_w = 0 ; 
 
        G4Opticks::Get()->collectHit(
             pos.x()/mm, 
             pos.y()/mm, 
             pos.z()/mm,
             time/ns,

             dir.x(),
             dir.y(),
             dir.z(),
             weight, 

             pol.x(),
             pol.y(),
             pol.z(),
             wavelength/nm, 

             flags_x,
             flags_y,
             flags_z,
             flags_w
        );
    }
#endif
 
    OpHit* hit = new OpHit ; 
    hit->ene = ene ;  
    hit->tim = time ;  
    hit->pos = pos ;  
    hit->dir = dir ;  
    hit->pol = pol ;  

    OpHitCollection* hc = pos.x() > 0 ? hitCollectionA : hitCollectionB ; 
    hc->insert(hit); 

    return true ; 
}


/**
SensitiveDetector::Initialize
------------------------------

* two example hit collections are instanciated and added to G4HCofThisEvent* HCE 

Invoked at event level, by::

    G4SDStructure::Initialize
    G4SDManager::PrepareNewEvent
    G4EventManager::DoProcessing
    G4EventManager::ProcessOneEvent
    G4RunManager::ProcessOneEvent
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn
    G4::beamOn


**/

void SensitiveDetector::Initialize(G4HCofThisEvent* HCE)   
{
    G4cout
        << "SensitiveDetector::Initialize"
        << " HCE " << HCE
        << " HCE.Capacity " << HCE->GetCapacity()
        << " SensitiveDetectorName " << SensitiveDetectorName
        << " collectionName[0] " << collectionName[0] 
        << " collectionName[1] " << collectionName[1] 
        << G4endl  
        ; 
 
    hitCollectionA = new OpHitCollection(SensitiveDetectorName,collectionName[0]);
    hitCollectionB = new OpHitCollection(SensitiveDetectorName,collectionName[1]);

    G4SDManager* SDMan = G4SDManager::GetSDMpointerIfExist() ;
    assert( SDMan ) ;  

    int hcid_A = SDMan->GetCollectionID(hitCollectionA);
    HCE->AddHitsCollection(hcid_A, hitCollectionA ); 

    int hcid_B = SDMan->GetCollectionID(hitCollectionB);
    HCE->AddHitsCollection(hcid_B, hitCollectionB ); 

    G4VHitsCollection* hcA = HCE->GetHC(hcid_A); 
    assert( hcA == hitCollectionA ); 

    G4VHitsCollection* hcB = HCE->GetHC(hcid_B); 
    assert( hcB == hitCollectionB ); 
}


/**
SensitiveDetector::EndOfEvent
-------------------------------

Invoked at event level by::

    G4SDStructure::Terminate
    G4SDManager::TerminateCurrentEvent
    G4EventManager::DoProcessing 
    G4EventManager::ProcessOneEvent
    G4RunManager::ProcessOneEvent
    G4RunManager::DoEventLoop
    G4RunManager::BeamOn

**/

void SensitiveDetector::EndOfEvent(G4HCofThisEvent* HCE) 
{
    G4cout
        << "SensitiveDetector::EndOfEvent"
        << " HCE " << HCE
        << " hitCollectionA->entries() " << hitCollectionA->entries()
        << " hitCollectionB->entries() " << hitCollectionB->entries()
        << " A+B " << hitCollectionA->entries() + hitCollectionB->entries()
        << " m_hit_count " << m_hit_count 
        << G4endl  
        ; 
}

/**
SensitiveDetector::DumpHitCollections
--------------------------------------

**/

void SensitiveDetector::DumpHitCollections(G4HCofThisEvent* HCE) // static
{
    G4SDManager* SDMan = G4SDManager::GetSDMpointerIfExist() ;
    assert( SDMan ) ;  

    G4HCtable* tab = SDMan->GetHCtable();
    for(G4int i=0 ; i < tab->entries() ; i++ )
    {   
        std::string sdName = tab->GetSDname(i);  
        std::string colName = tab->GetHCname(i);  
        std::string query = sdName + "/" + colName ; 
        int hcid = tab->GetCollectionID(query);
        OpHitCollection* hc = dynamic_cast<OpHitCollection*>(HCE->GetHC(hcid));
        if( hc == NULL ) continue ;  

        G4cout 
            << "SensitiveDetector::DumpHitCollections"
            << " query " << std::setw(20) << query 
            << " hcid "  << std::setw(4) << hcid 
            << " hc "    << hc
            << " hc.entries "    << hc->entries()
            << G4endl 
            ; 
    }
}

/**
SensitiveDetector::GetHitCollection
-------------------------------------

Static method giving access to hit collections from outside 
the normal ProceesHit machinery, eg from EventAction 

Hmm ... could just accept arrays of Opticks Hits here ?

**/

OpHitCollection* SensitiveDetector::GetHitCollection( G4HCofThisEvent* HCE, const char* query ) // static
{
    G4SDManager* SDMan = G4SDManager::GetSDMpointerIfExist() ;
    assert( SDMan ) ;  
    G4HCtable* tab = SDMan->GetHCtable();
    int hcid = tab->GetCollectionID(query);
    OpHitCollection* hc = dynamic_cast<OpHitCollection*>(HCE->GetHC(hcid));
    assert(hc); 
    return hc ;  
} 

