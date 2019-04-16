#pragma once

#include "X4_API_EXPORT.hh"
#include <string>

class G4VSolid ;
class X4GDMLWriteStructure ;  

/**
X4GDMLParser
=============

g4-;g4-cls G4GDMLParser

**/

#include "G4String.hh"
#include "G4GDMLParser.hh"

class X4_API X4GDMLParser  
{
    public:
        static void Write( const G4VSolid* solid, const char* path=NULL );  // NULL path writes to stdout
        static std::string ToString( const G4VSolid* solid ) ;  
    private:
        X4GDMLParser() ; 
        void write(const G4VSolid* solid, const char* path=NULL );
        std::string to_string( const G4VSolid* solid );
    private:
        X4GDMLWriteStructure* writer ;  

};

