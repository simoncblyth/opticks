
#include "SGDML.hh"
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

    // cannot skip and reuse existing despite it having the same digest 
    // as the pointer locations will differ so all the names will be different
    // relative to those in lv2sd for example
    if(exists) 
    {
        BFile::RemoveFile( path ) ; 
    }

    bool create = true ; 
    BFile::preparePath( path, create ) ;   

    LOG(info) << "export to " << path ; 

    G4GDMLParser* gdml = new G4GDMLParser ;
    G4bool refs = true ;
    G4String schemaLocation = "" ; 

    gdml->Write(path, world, refs, schemaLocation );
}


// based on G4GDMLWrite::GenerateName 
std::string CGDML::GenerateName(const char* name, const void* const ptr, bool addPointerToName )
{
    return SGDML::GenerateName(name, ptr, addPointerToName );
}



