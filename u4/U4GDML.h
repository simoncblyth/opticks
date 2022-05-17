#pragma once

class G4VPhysicalVolume ; 

struct U4GDML
{
    static G4VPhysicalVolume* Parse(const char* path);
};

#include "G4GDMLParser.hh"

inline G4VPhysicalVolume* U4GDML::Parse(const char* path)
{
    G4GDMLParser parser ; 

    parser.SetStripFlag(false); 

    bool validate = false ; 
    parser.Read(path, validate); 

    const G4String setupName = "Default" ;
    return parser.GetWorldVolume(setupName) ; 
}



