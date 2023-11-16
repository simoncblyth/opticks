#pragma once

#include "plog/Severity.h"
class G4VPhysicalVolume ; 
class G4GDMLParser ; 

struct U4GDML
{
    static constexpr const plog::Severity LEVEL = info ; 
    static constexpr const char* DefaultGDMLPath = "$UserGEOMDir/origin.gdml" ;  

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





#include "sdirect.h"
#include "sdirectory.h"
#include "spath.h"
#include "sstr.h"
#include "ssys.h"

#include "G4GDMLParser.hh"
#include "GDXML.hh"


/**
U4GDML::Read 
-------------

When geometry persisting is enabled the origin.gdml should be 
included in the output folder. Note that this might need
to have been GDML kludged to allow Geant4 for load it. 
For GDML kludging see the gdxml package which loads GDML using XercesC and does 
GDML fixups that allow Geant4 to parse the JUNO GDML. 

**/

inline const G4VPhysicalVolume* U4GDML::Read()
{
    return Read(DefaultGDMLPath); 
}

inline const G4VPhysicalVolume* U4GDML::Read(const char* path_)
{
    const char* path = path_ ? path_ : DefaultGDMLPath ; 
    bool exists = path ? spath::Exists(path) : false ; 
    LOG_IF(fatal, !exists) << " path invalid or does not exist [" << path << "]" ; 
    if(!exists) return nullptr ; 

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
    LOG_IF(error, world == nullptr) << " world NULL " ; 
    if(world == nullptr) return ; 

    U4GDML g(world) ; 
    g.write(path); 
}
inline void U4GDML::Write(const G4VPhysicalVolume* world, const char* base, const char* name )
{
    LOG_IF(error, world == nullptr) << " world NULL " ; 
    if(world == nullptr) return ; 

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
    const char* path = spath::Resolve(base, name); 
    read(path);  
}

/**
U4GDML::read
-------------

Attempt to swallow the G4cout "G4GDML:" logging using sdirect.h 
not working, something funny about G4cout ?::

   g4-cls G4ios

**/

inline void U4GDML::read(const char* path_)
{
    const char* path = spath::Resolve(path_); 

    parser->SetStripFlag(read_trim); 

    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {
        sdirect::cout_(coutbuf.rdbuf());
        sdirect::cerr_(cerrbuf.rdbuf());

        parser->Read(path, read_validate);  // noisy code 

    }
    std::string out = coutbuf.str();
    std::string err = cerrbuf.str();
    bool verbose = getenv("VERBOSE") != nullptr ;  
    std::cout << sdirect::OutputMessage("U4GDML::read", out, err, verbose );

    const G4String setupName = "Default" ;
    world = parser->GetWorldVolume(setupName) ; 
}

inline void U4GDML::write(const char* base, const char* name)
{
    const char* path = spath::Resolve(base, name); 
    write(path);  
}




/**
U4GDML::write
---------------

Example of steps taken when *path* is "/some/dir/to/example.gdml" 

1. rawpath "/some/dir/to/example_raw.gdml" is written using Geant4 GDML parser
2. rawpath GDML is read as XML and some issues may be fixed (using GDXML::Fix) 
3. fixed XML is written to original *path* "/some/dir/to/example.gdml"  

To disable use of GDXML::Fix define envvar::

    export U4GDML_GDXML_FIX_DISABLE=1

**/

inline void U4GDML::write(const char* path)
{
    assert( sstr::EndsWith(path, ".gdml") ); 
    const char* dstpath = path ; 

    const char* ekey = "U4GDML_GDXML_FIX_DISABLE" ; 
    bool U4GDML_GDXML_FIX_DISABLE = ssys::getenvbool(ekey) ;
    bool U4GDML_GDXML_FIX = !U4GDML_GDXML_FIX_DISABLE  ; 

    LOG(LEVEL) 
        << " ekey " << ekey 
        << " U4GDML_GDXML_FIX_DISABLE " << U4GDML_GDXML_FIX_DISABLE
        << " U4GDML_GDXML_FIX " << U4GDML_GDXML_FIX 
        ; 

    if(U4GDML_GDXML_FIX)
    {
        const char* rawpath = sstr::ReplaceEnd(path, ".gdml", "_raw.gdml" );  
        write_(rawpath); 
        LOG(LEVEL) << "[ Apply GDXML::Fix " << " rawpath " << rawpath << " dstpath " << dstpath ; 
        GDXML::Fix( dstpath, rawpath );     
        LOG(LEVEL) << "] Apply GDXML::Fix " << " rawpath " << rawpath << " dstpath " << dstpath ; 
    }
    else
    {
        LOG(LEVEL) << " NOT Applying GDXML::Fix " << " dstpath " << dstpath ; 
        write_(dstpath); 
    }
}

inline void U4GDML::write_(const char* path)
{
    LOG(LEVEL) << "[" ;    
    bool exists = spath::Exists(path) ; 
    int rc = exists ? spath::Remove(path) : 0 ; 

    LOG_IF(fatal, rc != 0 ) 
        << " FAILED TO REMOVE PATH [" << path << "]" 
        << " CHECK PERMISSIONS " 
        ; 

    LOG(LEVEL) 
        << " path " << ( path ? path : "-" ) 
        << " exists " << ( exists ? "YES" : "NO " )
        << " rc " << rc
        ;

    sdirectory::MakeDirsForFile(path,0);
    parser->Write(path, world, write_refs, write_schema_location); 
    LOG(LEVEL) << "]" ;    
}


