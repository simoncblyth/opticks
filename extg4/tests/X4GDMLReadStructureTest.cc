#include <fstream>

#include "OPTICKS_LOG.hh"
#include "BFile.hh"
#include "X4GDMLReadStructure.hh"

#include <xercesc/util/PlatformUtils.hpp>

const char* SOLID = R"LITERAL(

<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="SchemaLocation">
  <solids>
     <box lunit="mm" name="WorldBox0xc15cf400x3ebf070" x="4800000" y="4800000" z="4800000"/>
  </solid>
</gdml>

)LITERAL";

/**


**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    xercesc::XMLPlatformUtils::Initialize();

    const G4VSolid* solid = X4GDMLReadStructure::ReadSolidFromString(SOLID) ;  
    LOG(info) << " solid " << solid ; 

    G4cout << *solid ; 
    return 0 ; 
}


