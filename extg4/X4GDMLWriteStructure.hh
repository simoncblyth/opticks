#pragma once

#include "X4_API_EXPORT.hh"
#include <string>

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

        void write(const G4VSolid* solid, const char* path=NULL ); // to file or stdout when path is NULL

        std::string to_string( const G4VSolid* solid ); 
   private:
        void init();
        void add( const G4VSolid* solid ); 
        std::string write( const char* path=NULL ) ; 
  
   private:
        xercesc::DOMElement* gdml ; 
        xercesc::DOMImplementation* impl ;


};

