#include <vector>
#include <string>
#include <cstring>
#include <cassert>

#include "BStr.hh"
#include "BBnd.hh"

const char* BBnd::DuplicateOuterMaterial( const char* boundary0 )
{
    std::vector<std::string> elem ; 

    char delim = '/' ;

    BStr::split(elem, boundary0, delim );
    assert( elem.size() == 4 );

    std::string omat = elem[0] ; 
    //std::string osur = elem[1] ; 
    //std::string isur = elem[2] ; 
    std::string imat = elem[3] ; 

    assert( !omat.empty() );
    assert( !imat.empty() );

    std::vector<std::string> uelem ;  
    uelem.push_back( omat );
    uelem.push_back( "" );
    uelem.push_back( "" );
    uelem.push_back( omat );

    std::string ubnd = BStr::join(uelem, delim ); 

    return strdup(ubnd.c_str());
}


