#include "X4GDMLParser.hh"
#include "X4GDMLWriteStructure.hh"


X4GDMLParser::X4GDMLParser()
    :
    writer(new X4GDMLWriteStructure)
{
    xercesc::XMLPlatformUtils::Initialize();
}


/*
void X4GDMLParser::write( std::ostream& out,  const G4VSolid* solid)
{
}
*/

void X4GDMLParser::write(const G4String& filename, const G4VSolid* solid )
{
    writer->write( filename, solid ); 
}




