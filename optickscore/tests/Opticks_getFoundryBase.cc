
#include <cstdio>
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
 
    Opticks ok(argc, argv ); 

    ok.configure(); 

    const char* cfbase = ok.getFoundryBase() ; 

    printf("%s\n", cfbase ); 

    return 0 ; 
}
