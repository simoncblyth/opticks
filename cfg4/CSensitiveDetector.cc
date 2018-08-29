
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

    //LOG(info) << "." ; 

    return true ; 
}



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


