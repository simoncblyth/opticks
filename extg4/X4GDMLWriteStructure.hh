#pragma once

#include "X4_API_EXPORT.hh"

class G4VSolid ; 

/**
X4GDMLWrite
=============

g4-;g4-cls G4GDMLWrite


**/


#include "G4GDMLWriteStructure.hh"

class X4_API X4GDMLWriteStructure : public G4GDMLWriteStructure 
{
    public:
        X4GDMLWriteStructure() ; 
        //void write(const G4VSolid* solid); 

        void write(const G4String& filename, const G4VSolid* solid );


};

