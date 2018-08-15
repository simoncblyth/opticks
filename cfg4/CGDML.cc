
#include "CGDML.hh"
#include "BFile.hh"
#include "G4GDMLParser.hh"

#include "PLOG.hh"


void CGDML::Export(const char* dir, const char* name, const G4VPhysicalVolume* const world )
{
    std::string path = BFile::FormPath(dir, name);
    CGDML::Export( path.c_str(), world ); 
}

void CGDML::Export(const char* path, const G4VPhysicalVolume* const world )
{
    assert( world );

    bool exists = BFile::ExistsFile( path ); 
    if(exists) 
    {
        LOG(info) << "Skip export as GDML file exists already at " << path ; 
        return ; 
    }

    bool create = true ; 
    BFile::preparePath( path, create ) ;   

    LOG(info) << "export to " << path ; 

    G4GDMLParser* gdml = new G4GDMLParser ;
    G4bool refs = true ;
    G4String schemaLocation = "" ; 

    gdml->Write(path, world, refs, schemaLocation );
}

