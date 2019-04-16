#include "X4GDMLParser.hh"
#include "X4GDMLWriteStructure.hh"

void X4GDMLParser::Write( const G4VSolid* solid, const char* path, bool refs )  // static
{
    X4GDMLParser parser(refs) ; 
    parser.write(solid, path) ; 
}

std::string X4GDMLParser::ToString( const G4VSolid* solid, bool refs ) // static
{
    X4GDMLParser parser(refs) ; 
    return parser.to_string(solid); 
}

X4GDMLParser::X4GDMLParser(bool refs)
    :
    writer(NULL)
{
    xercesc::XMLPlatformUtils::Initialize();
    writer = new X4GDMLWriteStructure(refs) ; 
}

void X4GDMLParser::write(const G4VSolid* solid, const char* path )
{
    writer->write( solid, path ); 
}

std::string X4GDMLParser::to_string( const G4VSolid* solid )
{
    return writer->to_string(solid); 
}


