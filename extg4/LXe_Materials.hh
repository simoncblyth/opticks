
#pragma once

class G4Element ; 
class G4Material  ; 
class G4MaterialPropertiesTable ;

#include "X4_API_EXPORT.hh"

struct X4_API LXe_Materials
{
    LXe_Materials();

    //Materials & Elements
    G4Material* fLXe;
    G4Material* fAl;
    G4Element* fN;
    G4Element* fO;
    G4Material* fAir;
    G4Material* fVacuum;
    G4Element* fC;
    G4Element* fH;
    G4Material* fGlass;
    G4Material* fPstyrene;
    G4Material* fPMMA;
    G4Material* fPethylene1;
    G4Material* fPethylene2;

    G4MaterialPropertiesTable* fLXe_mt;
    G4MaterialPropertiesTable* fMPTPStyrene;




};


