#pragma once
/**
U4NistManager.h
================

Find names::

    g4-;g4-cc FindOrBuildMaterial
    g4-cls G4NistMaterialBuilder



**/
#include "G4NistManager.hh"

struct U4NistManager
{
    static G4Material* GetMaterial(const char* name) ; 
};

inline G4Material* U4NistManager::GetMaterial(const char* name)
{
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* material  = nist->FindOrBuildMaterial(name);
    return material ; 
}

