#pragma once
/**
U4SensitiveDetector.h
======================





**/

#include "G4VSensitiveDetector.hh"

struct U4SensitiveDetector : public G4VSensitiveDetector
{
    U4SensitiveDetector(const char* name); 

    G4bool ProcessHits(G4Step* step, G4TouchableHistory* hist);     
};

U4SensitiveDetector::U4SensitiveDetector(const char* name)
   :
   G4VSensitiveDetector(name)
{
}

G4bool U4SensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory* hist)
{
    //std::cout << "U4SensitiveDetector::ProcessHits" << std::endl ; 
    return true ; 
}

