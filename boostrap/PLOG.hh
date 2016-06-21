#include <plog/Severity.h>

// translate from boost log levels to plog 
using plog::fatal ;
using plog::error ;
using plog::warning ;
using plog::info ;
using plog::debug ;
using plog::verbose ;

// hmm dangerous but what alternative 
#define trace plog::verbose 
#include "BRAP_API_EXPORT.hh"
struct BRAP_API PLOG 
{
    int   level ; 
    PLOG(int argc, char** argv);
};

