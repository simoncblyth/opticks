#pragma once

#include "plog/Severity.h"
class G4VPhysicalVolume ; 
class G4GDMLParser ; 

struct U4GDML
{
    static plog::Severity LEVEL ; 
    static const G4VPhysicalVolume* Read();
    static const G4VPhysicalVolume* Read(const char* path);
    static const G4VPhysicalVolume* Read(const char* base, const char* name);
    static void Write(const G4VPhysicalVolume* world, const char* path);
    static void Write(const G4VPhysicalVolume* world, const char* base, const char* name) ;

    bool read_trim ; 
    bool read_validate ; 
    bool write_refs ; 
    const char* write_schema_location ; 

    U4GDML(const G4VPhysicalVolume* world_=nullptr ); 

    G4GDMLParser*      parser ;
    const G4VPhysicalVolume* world ;  

    void read( const char* base, const char* name);
    void read( const char* path);
    void write(const char* base, const char* name);
    void write(const char* path);
    void write_(const char* path);
};



//#include "PLOG.hh"
#include "SPath.hh"
#include "SStr.hh"
#include "G4GDMLParser.hh"
#include "GDXML.hh"


/**
U4GDML::Read 
-------------

TODO: the default path here needs to be standardly written when doing the Geant4 to Opticks translation 
in order for the origin GDML to always be available for tests. 
For GDML kludging see the gdxml package which loads GDML using XercesC and does 
GDML fixups that allow Geant4 to parse the JUNO GDML. 

**/

inline const G4VPhysicalVolume* U4GDML::Read()
{
    return Read("$IDPath/origin_GDMLKludge.gdml"); 
}

inline const G4VPhysicalVolume* U4GDML::Read(const char* path)
{
    U4GDML g ; 
    g.read(path); 
    return g.world ; 
}
inline const G4VPhysicalVolume* U4GDML::Read(const char* base, const char* name)
{
    U4GDML g ; 
    g.read(base, name); 
    return g.world ; 
}
inline void U4GDML::Write(const G4VPhysicalVolume* world, const char* path)
{
    U4GDML g(world) ; 
    g.write(path); 
}
inline void U4GDML::Write(const G4VPhysicalVolume* world, const char* base, const char* name )
{
    U4GDML g(world) ; 
    g.write(base, name); 
}



inline U4GDML::U4GDML(const G4VPhysicalVolume* world_)
    :
    read_trim(false),
    read_validate(false),
    write_refs(true),
    write_schema_location(""),
    parser(new G4GDMLParser),
    world(world_)
{
}

inline void U4GDML::read(const char* base, const char* name)
{
    const char* path = SPath::Resolve(base, name, NOOP); 
    read(path);  
}
inline void U4GDML::read(const char* path_)
{
    const char* path = SPath::Resolve(path_, NOOP); 

    parser->SetStripFlag(read_trim); 
    parser->Read(path, read_validate); 
    const G4String setupName = "Default" ;
    world = parser->GetWorldVolume(setupName) ; 
}

inline void U4GDML::write(const char* base, const char* name)
{
    const char* path = SPath::Resolve(base, name, FILEPATH); 
    write(path);  
}

/**
U4GDML::write
---------------

Example of steps taken when *path* is "/some/dir/to/example.gdml" 

1. rawpath "/some/dir/to/example_raw.gdml" is written using Geant4 GDML parser
2. rawpath GDML is read as XML and some issues may be fixed (using GDXML::Fix) 
3. fixed XML is written to original *path* "/some/dir/to/example.gdml"  

**/

inline void U4GDML::write(const char* path)
{
    assert( SStr::EndsWith(path, ".gdml") ); 
    const char* rawpath = SStr::ReplaceEnd(path, ".gdml", "_raw.gdml" );  
    write_(rawpath); 

    const char* srcpath = rawpath ; 
    const char* dstpath = path ; 
    GDXML::Fix( dstpath, srcpath );     
}

inline void U4GDML::write_(const char* path)
{
    if(SPath::Exists(path)) SPath::Remove(path); 
    parser->Write(path, world, write_refs, write_schema_location); 
}


