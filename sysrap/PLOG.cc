#include <cstring>
#include "PLOG.hh"

PLOG::PLOG(int argc, char** argv)  : level(info)
{
    const char* ll = "INFO" ;
    for(int i=1 ; i < argc ; ++i )
    {
        if(strcmp(argv[i], "--trace")==0)   ll = "VERBOSE" ; 
        if(strcmp(argv[i], "--verbose")==0) ll = "VERBOSE" ; 
        if(strcmp(argv[i], "--debug")==0)   ll = "DEBUG" ; 
        if(strcmp(argv[i], "--info")==0)    ll = "INFO" ; 
        if(strcmp(argv[i], "--warning")==0) ll = "WARNING" ; 
        if(strcmp(argv[i], "--error")==0)   ll = "ERROR" ; 
        if(strcmp(argv[i], "--fatal")==0)   ll = "FATAL" ; 
    }
    plog::Severity severity = plog::severityFromString(ll) ;
    level = static_cast<int>(severity); 
}

