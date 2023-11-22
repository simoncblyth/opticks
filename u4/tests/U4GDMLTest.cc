/**
U4GDMLTest.cc
===============

**/

#include "spath.h"
#include "OPTICKS_LOG.hh"
#include "U4GDML.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* srcpath = argc > 1 ? argv[1] :  nullptr ; 
    if(srcpath == nullptr) return 0 ; 

    const char* dstpath = spath::Resolve("$TMP/U4GDMLTest/out.gdml") ; 
    sdirectory::MakeDirsForFile(dstpath); 

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

