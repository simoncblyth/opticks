/**
This failed to reproduce a problem of mangled LV names 
seen in the switch to 10.4.2 see: 

* notes/issues/Geant4_update_to_10_4_2.rst

But it did reveal interference between some PLOG.hh dangerous 
define of trace and xercesc headers, see:

* notes/issues/PLOG_dangerous_trace_define.rst

And fixing that issue fixed the mangled names problem, so 
this was a useful check even though it didnt reproduce the problem

**/


#include "DetectorConstruction.hh"

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"

#include "OPTICKS_LOG.hh"
#include "G4GDMLParser.hh"

void write_gdml( const G4VPhysicalVolume* pv, const char* path )
{
    LOG(info) << "export to " << path ; 

    G4GDMLParser gdml ;
    G4bool refs = true ;
    G4String schemaLocation = "" ; 

    gdml.Write(path, pv, refs, schemaLocation );
}

G4VPhysicalVolume* read_gdml( const char* path )
{
    G4GDMLParser gdml ;
    bool validate = false ; 
    bool trimPtr = false ; 
    gdml.SetStripFlag(trimPtr);
    gdml.Read(path, validate);
    return gdml.GetWorldVolume() ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    DetectorConstruction dc ;  

    G4VPhysicalVolume* pv = dc.Construct() ; 
    G4LogicalVolume* lv = pv->GetLogicalVolume(); 

    LOG(info) << " pv " << pv->GetName() ; 
    LOG(info) << " lv " << lv->GetName() ; 

    const char* path = "/tmp/GDMLMangledLVNames.gdml" ; 

    write_gdml( pv, path ); 


    G4VPhysicalVolume* pv2 = read_gdml( path ) ;
    G4LogicalVolume* lv2 = pv2->GetLogicalVolume(); 

    LOG(info) << " pv2 " << pv2->GetName() ; 
    LOG(info) << " lv2 " << lv2->GetName() ; 

    return 0 ; 
}


