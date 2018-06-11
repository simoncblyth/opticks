#pragma once

#include "X4_API_EXPORT.hh"
#include <vector>
#include <string>

class G4VSolid ; 

/**
X4SolidList
=============

Unlike most X4 classes there is no directly corresponding G4 class
which is converted from. G4SolidStore is somewhat related.

X4SolidList is used from the X4PhysicalVolume TraverseVolumeTree
structure traverse to collect G4VSolid instances.

**/

class X4_API X4SolidList
{
    public:
        X4SolidList(); 
        void addSolid(G4VSolid* solid); 
        bool hasSolid(G4VSolid* solid) const ;
        std::string desc() const ; 
        unsigned getNumSolids() const ; 
    private:
        std::vector<G4VSolid*> m_solidlist ; 
};

