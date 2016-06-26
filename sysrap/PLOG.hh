
// NB DANGEROUS DEFINES : INCLUDE AS THE LAST HEADER
//    OTHERWISE RANDOMLY REPLACES STRINGS IN SYSTEM HEADERS

#include <cstddef>
#include <plog/Log.h>

// translate from boost log levels to plog 
using plog::fatal ;
using plog::error ;
using plog::warning ;
using plog::info ;
using plog::debug ;
using plog::verbose ;

// hmm dangerous but what alternative 
#define trace plog::verbose 

#include "SYSRAP_API_EXPORT.hh"

struct PLOG ; 

struct SYSRAP_API PLOG 
{
    int    argc ; 
    char** argv ;
    int   level ; 

    PLOG(int argc, char** argv, const char* fallback="VERBOSE", const char* prefix=NULL );

    const char* name(); 
    int parse( const char* fallback);
    int parse( plog::Severity _fallback);
    int prefix_parse( const char* fallback, const char* prefix);
    int prefix_parse( plog::Severity _fallback, const char* prefix);

    static int  _parse(int argc, char** argv, const char* fallback);
    static int  _prefix_parse(int argc, char** argv, const char* fallback, const char* prefix);
    static void _dump(const char* msg, int argc, char** argv);
    static const char* _name(plog::Severity severity);
    static const char* _name(int level);

    static PLOG* instance ; 
};


#include "PLOG_INIT.hh"


