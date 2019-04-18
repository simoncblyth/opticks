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
        static const char* PreparePath( const char* prefix, int lvidx, const char* ext=".gdml"  ); 
    public:
        static void Write( const G4VSolid* solid, const char* path, bool refs );  // NULL path writes to stdout
        static std::string ToString( const G4VSolid* solid, bool refs ) ;  
    private:
        X4GDMLParser(bool refs) ; 
        void write(const G4VSolid* solid, const char* path);
        std::string to_string( const G4VSolid* solid);
    private:
        X4GDMLWriteStructure* writer ;  

};

