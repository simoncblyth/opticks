#include <fstream>

#include "OPTICKS_LOG.hh"
#include "BFile.hh"
#include "X4Dump.hh"
#include "X4GDMLReadStructure.hh"

#include <xercesc/util/PlatformUtils.hpp>

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* path = argc > 1 ? argv[1] : NULL ; 
    if(path == NULL) LOG(fatal) << "expecting argument with path of gdml file to parse"  ; 
    if(path == NULL) return 0 ; 

    xercesc::XMLPlatformUtils::Initialize();

    X4GDMLReadStructure reader ; 
    reader.readFile(path) ; 
    reader.dumpMatrixMap("X4GDMLReadStructure2Test");

    X4Dump::G4("mt,sk,bs");  

    return 0 ; 
}


