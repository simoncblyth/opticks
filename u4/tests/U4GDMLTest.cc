/**
U4GDMLTest.cc
===============

::

    epsilon:u4 blyth$ GEOM=J006 U4GDMLTest 
    G4GDML: Reading '/Users/blyth/.opticks/GEOM/J006/origin.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/Users/blyth/.opticks/GEOM/J006/origin.gdml' done!
    G4GDMLParser::read             yielded chars :  cout      0 cerr      0 : set VERBOSE to see them 
    U4GDML::write@186:  ekey U4GDML_GDXML_FIX_DISABLE U4GDML_GDXML_FIX_DISABLE 0 U4GDML_GDXML_FIX 1
    G4GDML: Writing '/tmp/blyth/opticks/U4GDMLTest/out_raw.gdml'...
    G4GDML: Writing definitions...
    ...

**/


#include "SPath.hh"
#include "OPTICKS_LOG.hh"
#include "U4GDML.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* srcpath = argc > 1 ? argv[1] :  nullptr ; 
    const char* dstpath = SPath::Resolve("$TMP/U4GDMLTest/out.gdml", FILEPATH) ; 

    const G4VPhysicalVolume* world = U4GDML::Read(srcpath) ;  

    U4GDML::Write(world, dstpath ); 

    LOG(info) 
        << " argv[0] " << argv[0] << std::endl 
        << " srcpath " << ( srcpath ? srcpath : "-" ) << std::endl 
        << " dstpath " << ( dstpath ? dstpath : "-" )  << std::endl 
        << " world " << ( world ? "Y" : "N" )  << std::endl  
        ;

    return 0 ; 

}
