#include "OPTICKS_LOG.hh"
#include "X4MaterialWaterStandalone.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    X4MaterialWaterStandalone mws ; 
    mws.dump(); 

    return 0 ; 
}


