// TEST=SEnvTest om-t

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "OPTICKS_LOG.hh"

/**
SEnvTest
=========

Dump envvars matching the prefix, or all when no prefix.
Used for checking "fabricated" ctest, ie ctest without its own 
executable, but instead with its own environment. 

**/

int main(int argc, char** argv, char** envp)
{
    OPTICKS_LOG(argc, argv) ;

    const char* prefix = argc > 1 ? argv[1] : NULL ; 

    LOG(debug) << " prefix " << ( prefix ? prefix : "NONE" ); 

    while(*envp)
    {
        if(prefix != NULL && strncmp(*envp, prefix, strlen(prefix)) == 0) 
        { 
            LOG(info) << *envp ;  
        } 
        envp++ ; 
    }

    return 0 ; 
}
