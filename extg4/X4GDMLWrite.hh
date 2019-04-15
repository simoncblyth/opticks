#pragma once

#include "X4_API_EXPORT.hh"

class G4VSolid ; 

/**
X4GDMLWrite
========

**/


#include "G4GDMLWrite.hh"

class X4_API X4GDMLWrite : public G4GDMLWrite 
{
    public:
        X4GDMLWrite() ; 
        void write(const G4VSolid* solid); 

};

