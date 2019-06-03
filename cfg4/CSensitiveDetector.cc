#include <csignal>

#include "CHit.hh"
#include "CSensitiveDetector.hh"
#include "G4OpticalPhoton.hh"
#include "G4SDManager.hh"
#include "PLOG.hh"


const char* CSensitiveDetector::SDName = NULL ; 
const char* CSensitiveDetector::collectionNameA = "OpHitCollectionA" ;
const char* CSensitiveDetector::collectionNameB = "OpHitCollectionB" ;

CSensitiveDetector::CSensitiveDetector(const char* name) 
    :
    G4VSensitiveDetector(name)
{
    SDName = strdup(name) ; 
    collectionName.insert(collectionNameA); 
    collectionName.insert(collectionNameB); 

    G4SDManager* SDMan = G4SDManager::GetSDMpointer() ;
    SDMan->AddNewDetector(this); 
}

G4bool CSensitiveDetector::ProcessHits(G4Step* step,G4TouchableHistory* )
{
    G4Track* track = step->GetTrack();
    if (track->GetDefinition() != G4OpticalPhoton::Definition()) return false ; 

    G4double ene = step->GetTotalEnergyDeposit();
    G4StepPoint* point = step->GetPreStepPoint();
    G4double tim = point->GetGlobalTime(); 
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();
 
    CHit* hit = new CHit ; 
    hit->ene = ene ;  
    hit->tim = tim ;  
    hit->pos = pos ;  
    hit->dir = dir ;  
    hit->pol = pol ;  

    CHitCollection* hc = pos.x() > 0 ? hitCollectionA : hitCollectionB ; 
    hc->insert(hit); 

    //LOG(info) ; 
    //std::raise(SIGINT);


    return true ; 
}

/*
(gdb) bt
#0  0x00007fffe2035207 in raise () from /usr/lib64/libc.so.6
#1  0x00007fffe20368f8 in abort () from /usr/lib64/libc.so.6
#2  0x00007fffe202e026 in __assert_fail_base () from /usr/lib64/libc.so.6
#3  0x00007fffe202e0d2 in __assert_fail () from /usr/lib64/libc.so.6
#4  0x00007fffefd6d1b3 in CSensitiveDetector::ProcessHits (this=0x8ef800, step=0x88a800) at /home/blyth/opticks/cfg4/CSensitiveDetector.cc:49
#5  0x00007fffec12d431 in G4VSensitiveDetector::Hit (this=0x8ef800, aStep=0x88a800) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/digits_hits/detector/include/G4VSensitiveDetector.hh:122
#6  0x00007fffec12b6df in G4SteppingManager::Stepping (this=0x88a660) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/tracking/src/G4SteppingManager.cc:237
#7  0x00007fffec137236 in G4TrackingManager::ProcessOneTrack (this=0x88a620, apValueG4Track=0x2243bd0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/tracking/src/G4TrackingManager.cc:126
#8  0x00007fffec3afd46 in G4EventManager::DoProcessing (this=0x88a590, anEvent=0x216f2a0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:185
#9  0x00007fffec3b0572 in G4EventManager::ProcessOneEvent (this=0x88a590, anEvent=0x216f2a0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:338
#10 0x00007fffec6b2665 in G4RunManager::ProcessOneEvent (this=0x701520, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:399
#11 0x00007fffec6b24d7 in G4RunManager::DoEventLoop (this=0x701520, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
#12 0x00007fffec6b1d2d in G4RunManager::BeamOn (this=0x701520, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
#13 0x00007fffefded44d in CG4::propagate (this=0x708570) at /home/blyth/opticks/cfg4/CG4.cc:331
#14 0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffd280) at /home/blyth/opticks/okg4/OKG4Mgr.cc:177
#15 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffd280) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
#16 0x00000000004039a7 in main (argc=7, argv=0x7fffffffd5b8) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
(gdb) 
*/



void CSensitiveDetector::Initialize(G4HCofThisEvent* HCE)
{
    LOG(info) 
        << " HCE " << HCE
        << " HCE.Capacity " << HCE->GetCapacity()
        << " SensitiveDetectorName " << SensitiveDetectorName
        << " collectionName[0] " << collectionName[0] 
        << " collectionName[1] " << collectionName[1] 
        ; 

    hitCollectionA = new CHitCollection(SensitiveDetectorName,collectionName[0]);
    hitCollectionB = new CHitCollection(SensitiveDetectorName,collectionName[1]);

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

void CSensitiveDetector::EndOfEvent(G4HCofThisEvent* HCE)
{
    LOG(info) 
        << " HCE " << HCE
        << " hitCollectionA->entries() " << hitCollectionA->entries()
        << " hitCollectionB->entries() " << hitCollectionB->entries()
        ; 
}



void CSensitiveDetector::DumpHitCollections(G4HCofThisEvent* HCE) // static
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
        CHitCollection* hc = dynamic_cast<CHitCollection*>(HCE->GetHC(hcid));
        if( hc == NULL ) continue ;  

        LOG(info) 
            << " query " << std::setw(20) << query 
            << " hcid "  << std::setw(4) << hcid 
            << " hc "    << hc
            << " hc.entries "    << hc->entries()
            ; 
    }
}


/**
CSensitiveDetector::GetHitCollection
--------------------------------------

This is a static method enabling access to hit collections from outside 
the normal ProceesHit machinery, eg from EventAction 

Hmm ... could just accept arrays of Opticks Hits here  

**/

CHitCollection* CSensitiveDetector::GetHitCollection( G4HCofThisEvent* HCE, const char* query ) // static
{

    G4SDManager* SDMan = G4SDManager::GetSDMpointerIfExist() ;
    assert( SDMan ) ;  
    G4HCtable* tab = SDMan->GetHCtable();
    int hcid = tab->GetCollectionID(query);
    CHitCollection* hc = dynamic_cast<CHitCollection*>(HCE->GetHC(hcid));
    assert(hc); 
    return hc ;  
} 


