#pragma once

#include "plog/Severity.h"
class G4VPhysicalVolume ; 
class G4GDMLParser ; 

struct U4GDML
{
    static plog::Severity LEVEL ; 
    static G4VPhysicalVolume* Read();
    static G4VPhysicalVolume* Read(const char* path);
    static void Write(G4VPhysicalVolume* world, const char* path);

    bool read_trim ; 
    bool read_validate ; 
    bool write_refs ; 
    const char* write_schema_location ; 

    U4GDML(G4VPhysicalVolume* world_=nullptr ); 

    G4GDMLParser*      parser ;
    G4VPhysicalVolume* world ;  

    void read( const char* path);
    void write(const char* path);
};



//#include "PLOG.hh"
#include "SPath.hh"
#include "G4GDMLParser.hh"


/**
U4GDML::Read 
-------------

TODO: the default path here needs to be standardly written when doing the Geant4 to Opticks translation 
in order for the origin GDML to always be available for tests. 
For GDML kludging see the gdxml package which loads GDML using XercesC and does 
GDML fixups that allow Geant4 to parse the JUNO GDML. 

**/

inline G4VPhysicalVolume* U4GDML::Read()
{
    return Read("$IDPath/origin_GDMLKludge.gdml"); 
}

inline G4VPhysicalVolume* U4GDML::Read(const char* path)
{
    U4GDML g ; 
    g.read(path); 
    return g.world ; 
}
inline void U4GDML::Write(G4VPhysicalVolume* world, const char* path)
{
    U4GDML g(world) ; 
    g.write(path); 
}

inline U4GDML::U4GDML(G4VPhysicalVolume* world_)
    :
    read_trim(false),
    read_validate(false),
    write_refs(true),
    write_schema_location(""),
    parser(new G4GDMLParser),
    world(world_)
{
}

inline void U4GDML::read(const char* path_)
{
    const char* path = SPath::Resolve(path_, NOOP); 

    parser->SetStripFlag(read_trim); 
    parser->Read(path, read_validate); 
    const G4String setupName = "Default" ;
    world = parser->GetWorldVolume(setupName) ; 
}

inline void U4GDML::write(const char* path)
{
    if(SPath::Exists(path)) 
    {
        SPath::Remove(path); 
    }
    parser->Write(path, world, write_refs, write_schema_location); 
}


