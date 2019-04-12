#include <string>
#include <cstring>
#include "SPath.hh"

const char* SPath::Stem( const char* name )
{
    std::string arg = name ;
    std::string base = arg.substr(0, arg.find_last_of(".")) ; 
    return strdup( base.c_str() ) ; 
}

