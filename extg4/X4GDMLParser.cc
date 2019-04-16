#include "X4GDMLParser.hh"
#include "X4GDMLWriteStructure.hh"



void X4GDMLParser::Write( const G4VSolid* solid, const char* path )  // static
{
    X4GDMLParser parser ; 
    parser.write(solid, path) ; 
}

std::string X4GDMLParser::ToString( const G4VSolid* solid ) // static
{
    X4GDMLParser parser ; 
    return parser.to_string(solid); 
}
 


X4GDMLParser::X4GDMLParser()
    :
    writer(NULL)
{
    xercesc::XMLPlatformUtils::Initialize();
    writer = new X4GDMLWriteStructure ; 
}

void X4GDMLParser::write(const G4VSolid* solid, const char* path )
{
    writer->write( solid, path ); 
}

std::string X4GDMLParser::to_string( const G4VSolid* solid )
{
    return writer->to_string(solid); 
}


