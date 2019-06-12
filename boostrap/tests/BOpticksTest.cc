
#include "BOpticks.hh"
#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    BOpticks ok(argc, argv, "--envkey" );  
    if(ok.getError() > 0) return 0 ;  

    const char* path = ok.getPath() ; 
    LOG(info) << path ; 

    return 0 ; 
}

