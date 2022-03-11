
#include "OPTICKS_LOG.hh"
#include "SOpticks.hh"

int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv); 

    const char* argforced = nullptr ; 
    SOpticks ok(argc, argv, argforced ); 

    LOG(info) << " ok.hasArg(\"--hello\") " << ok.hasArg("--hello") ;  

    return 0 ; 
}
