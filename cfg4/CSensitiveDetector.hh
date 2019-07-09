#pragma once

#include "G4VSensitiveDetector.hh"
#include "CHit.hh"
#include "plog/Severity.h"

class G4Step ; 
class G4TouchableHistory ;
class G4HCofThisEvent ; 

/**
CSensitiveDetector
====================

Canonical m_sd instance is ctor resident of CG4 

**/

class CSensitiveDetector : public G4VSensitiveDetector
{
    public:
        static const plog::Severity LEVEL ;  
        static CHitCollection* GetHitCollection(G4HCofThisEvent* HCE, const char* query );
        static void DumpHitCollections(G4HCofThisEvent* HCE);

        static const char* SDName ;
        static const char* collectionNameA ;
        static const char* collectionNameB ;

    public:
        CSensitiveDetector(const char* name) ;  

        void Initialize(G4HCofThisEvent* HCE);
        void EndOfEvent(G4HCofThisEvent* HCE);

        G4bool ProcessHits(G4Step* aStep,G4TouchableHistory* ROhist);

    private:
        CHitCollection* hitCollectionA ; 
        CHitCollection* hitCollectionB ; 

};




