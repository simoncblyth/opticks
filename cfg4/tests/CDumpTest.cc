#include "OKConf.hh"
#include "OPTICKS_LOG.hh"
#include "CGDML.hh"
#include "CDump.hh"

/**
CDumpTest
==========

Parse GDML and dump material and surface properties controlled by config string.::

    CDumpTest /tmp/v1.gdml mt,sk,bs

        mt:material
        sk:skinsurface 
        bs:bordersurface

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << "OKConf::Geant4VersionInteger() : " << OKConf::Geant4VersionInteger()  ;

    const char* path = argc > 1 ? argv[1] : NULL  ; 
    const char* cfg  = argc > 2 ? argv[2] : "mt,sk,bs" ; 

    if(!path) LOG(error) << " expecting first argument path to GDML " ; 
    if(!path) return 0 ; 

    LOG(info) << " parsing " << path ; 
    G4VPhysicalVolume* world = CGDML::Parse(path); 
    assert( world ); 

    CDump::G4(cfg); 

    LOG(info) << "OKConf::Geant4VersionInteger() : " << OKConf::Geant4VersionInteger()  ;
    return 0 ; 
}

// om-;TEST=CDumpTest;om-t 

