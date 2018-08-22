#pragma once

#include "G4VSensitiveDetector.hh"
#include "OpHit.hh"

class G4Step ; 
class G4TouchableHistory ;
class G4HCofThisEvent ; 


struct SensitiveDetector : public G4VSensitiveDetector
{
    static OpHitCollection* GetHitCollection(G4HCofThisEvent* HCE, const char* query );
    static void DumpHitCollections(G4HCofThisEvent* HCE);

    static const char* SDName ;
    static const char* collectionNameA ;
    static const char* collectionNameB ;

    SensitiveDetector(const char* name) ;  

    void Initialize(G4HCofThisEvent* HCE);
    void EndOfEvent(G4HCofThisEvent* HCE);
    G4bool ProcessHits(G4Step* aStep,G4TouchableHistory* ROhist);

    OpHitCollection* hitCollectionA ; 
    OpHitCollection* hitCollectionB ; 

    unsigned m_hit_count ;

};




