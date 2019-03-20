
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
 
        G4Opticks::GetOpticks()->collectHit(
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
#0  0x00007fffe00b5277 in raise () from /usr/lib64/libc.so.6
#1  0x00007fffe00b6968 in abort () from /usr/lib64/libc.so.6
#2  0x00007fffe00ae096 in __assert_fail_base () from /usr/lib64/libc.so.6
#3  0x00007fffe00ae142 in __assert_fail () from /usr/lib64/libc.so.6
#4  0x000000000041ac0b in SensitiveDetector::Initialize (this=0x8b8e80, HCE=0x1f08980) at /home/blyth/opticks/examples/Geant4/CerenkovMinimal/SensitiveDetector.cc:111
#5  0x00007ffff0b8f2d4 in G4SDStructure::Initialize (this=0x8ba890, HCE=0x1f08980) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/digits_hits/detector/src/G4SDStructure.cc:201
#6  0x00007ffff0b8d6ba in G4SDManager::PrepareNewEvent (this=0x8ba830) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/digits_hits/detector/src/G4SDManager.cc:112
#7  0x00007ffff396aae6 in G4EventManager::DoProcessing (this=0x8791b0, anEvent=0x1ecd1c0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:147
#8  0x00007ffff396b572 in G4EventManager::ProcessOneEvent (this=0x8791b0, anEvent=0x1ecd1c0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:338
#9  0x00007ffff3c6d665 in G4RunManager::ProcessOneEvent (this=0x6fe380, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:399
#10 0x00007ffff3c6d4d7 in G4RunManager::DoEventLoop (this=0x6fe380, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
#11 0x00007ffff3c6cd2d in G4RunManager::BeamOn (this=0x6fe380, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
#12 0x000000000041a420 in G4::beamOn (this=0x7fffffffd310, nev=1) at /home/blyth/opticks/examples/Geant4/CerenkovMinimal/G4.cc:62
#13 0x000000000041a2dd in G4::G4 (this=0x7fffffffd310, nev=1) at /home/blyth/opticks/examples/Geant4/CerenkovMinimal/G4.cc:50
#14 0x00000000004097fe in main (argc=1, argv=0x7fffffffd488) at /home/blyth/opticks/examples/Geant4/CerenkovMinimal/CerenkovMinimal.cc:7
(gdb) 
**/

void SensitiveDetector::Initialize(G4HCofThisEvent* HCE)   // invoked by G4EventManager::ProcessOneEvent/G4EventManager::DoProcessing/G4SDManager::PrepareNewEvent/G4SDStructure::Initialize
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
(gdb) bt
#0  0x00007fffe00b5277 in raise () from /usr/lib64/libc.so.6
#1  0x00007fffe00b6968 in abort () from /usr/lib64/libc.so.6
#2  0x00007fffe00ae096 in __assert_fail_base () from /usr/lib64/libc.so.6
#3  0x00007fffe00ae142 in __assert_fail () from /usr/lib64/libc.so.6
#4  0x000000000041b1a3 in SensitiveDetector::EndOfEvent (this=0x8bae80, HCE=0x1f0a560) at /home/blyth/opticks/examples/Geant4/CerenkovMinimal/SensitiveDetector.cc:162
#5  0x00007ffff0b8f3ec in G4SDStructure::Terminate (this=0x8bc890, HCE=0x1f0a560) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/digits_hits/detector/src/G4SDStructure.cc:211
#6  0x00007ffff0b8d706 in G4SDManager::TerminateCurrentEvent (this=0x8bc830, HCE=0x1f0a560) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/digits_hits/detector/src/G4SDManager.cc:118
#7  0x00007ffff396b153 in G4EventManager::DoProcessing (this=0x87b1b0, anEvent=0x1ec9d90) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:263
#8  0x00007ffff396b572 in G4EventManager::ProcessOneEvent (this=0x87b1b0, anEvent=0x1ec9d90) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:338
#9  0x00007ffff3c6d665 in G4RunManager::ProcessOneEvent (this=0x700380, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:399
#10 0x00007ffff3c6d4d7 in G4RunManager::DoEventLoop (this=0x700380, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
#11 0x00007ffff3c6cd2d in G4RunManager::BeamOn (this=0x700380, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
#12 0x000000000041a5e0 in G4::beamOn (this=0x7fffffffd310, nev=1) at /home/blyth/opticks/examples/Geant4/CerenkovMinimal/G4.cc:62
#13 0x000000000041a49d in G4::G4 (this=0x7fffffffd310, nev=1) at /home/blyth/opticks/examples/Geant4/CerenkovMinimal/G4.cc:50
#14 0x00000000004099be in main (argc=1, argv=0x7fffffffd488) at /home/blyth/opticks/examples/Geant4/CerenkovMinimal/CerenkovMinimal.cc:7
(gdb) 
**/

void SensitiveDetector::EndOfEvent(G4HCofThisEvent* HCE) // invoked by G4EventManager::ProcessOneEvent/G4EventManager::DoProcessing/G4SDManager::TerminateCurrentEvent/G4SDStructure::Terminate
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


OpHitCollection* SensitiveDetector::GetHitCollection( G4HCofThisEvent* HCE, const char* query ) // static
{
    /**
       This a static method enabling access to hit collections from outside 
       the normal ProceesHit machinery, eg from EventAction 

       Hmm ... could just accept arrays of Opticks Hits here  

    **/

    G4SDManager* SDMan = G4SDManager::GetSDMpointerIfExist() ;
    assert( SDMan ) ;  
    G4HCtable* tab = SDMan->GetHCtable();
    int hcid = tab->GetCollectionID(query);
    OpHitCollection* hc = dynamic_cast<OpHitCollection*>(HCE->GetHC(hcid));
    assert(hc); 
    return hc ;  
} 


