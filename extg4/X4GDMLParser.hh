#pragma once

#include "X4_API_EXPORT.hh"

class G4VSolid ;

class X4GDMLWriteStructure ;  

/**
X4GDMLParser
=============

g4-;g4-cls G4GDMLParser


**/

//#include <ostream>
#include "G4String.hh"
#include "G4GDMLParser.hh"

class X4_API X4GDMLParser  
{
    public:
        X4GDMLParser() ; 
/*
        void dump( const G4VSolid* solid);
        void write( std::ostream& out,  const G4VSolid* solid);
*/
        void write(const G4String& filename, const G4VSolid* solid );

    private:
        X4GDMLWriteStructure* writer ;  

};

