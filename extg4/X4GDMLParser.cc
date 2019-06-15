#include "SDirect.hh"
#include "BStr.hh"
#include "BFile.hh"

#include "X4GDMLParser.hh"
#include "X4GDMLWriteStructure.hh"

#include "PLOG.hh"

const char* X4GDMLParser::PreparePath( const char* prefix, int lvidx, const char* ext  ) // static
{ 
    //std::string dir = BFile::FormPath( prefix, "tests" ); 
    const char* x = lvidx < 0 ? "n" : "p" ; 
    std::string name = BStr::concat(x, BStr::utoa(lvidx < 0 ? -lvidx : lvidx, 3, true), ext ) ; 
    bool create = true ; 
    std::string path = BFile::preparePath( prefix, name.c_str(), create); 
    return strdup(path.c_str()); 
}

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

void X4GDMLParser::write_noisily(const G4VSolid* solid, const char* path )
{
    LOG(info) << "[" ; 
    writer->write( solid, path ); 
    LOG(info) << "]" ; 
}

/**
X4GDMLParser::write
---------------------

Stream redirection from SDirect avoids non-error output from G4GDMLParser

**/

void X4GDMLParser::write(const G4VSolid* solid, const char* path )
{
    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
       cout_redirect out(coutbuf.rdbuf());
       cerr_redirect err(cerrbuf.rdbuf());

       writer->write( solid, path ); 

    }   

    std::string cout_ = coutbuf.str() ; 
    std::string cerr_ = cerrbuf.str() ; 

    //if(cout_.size() > 0) LOG(info)    << "cout:" << cout_ ; 
    if(cerr_.size() > 0) LOG(error) << "cerr:"<< cerr_ ; 
}


std::string X4GDMLParser::to_string( const G4VSolid* solid )
{
    std::string gdml ; 

    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
       cout_redirect out(coutbuf.rdbuf());
       cerr_redirect err(cerrbuf.rdbuf());

       gdml = writer->to_string(solid); 

    }   
    std::string cout_ = coutbuf.str() ; 
    std::string cerr_ = cerrbuf.str() ; 

    //if(cout_.size() > 0) LOG(info)    << "cout:" << cout_ ; 
    if(cerr_.size() > 0) LOG(error) << "cerr:"<< cerr_ ; 
    return gdml ; 
}


