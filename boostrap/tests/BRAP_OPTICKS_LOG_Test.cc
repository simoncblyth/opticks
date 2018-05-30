#include "OPTICKS_LOG.hh"

int main(int argc , char** argv )
{
    OPTICKS_LOG_COLOR__(argc, argv) ; 

    OPTICKS_LOG::Check();

    LOG(info) << argv[0] ; 

    return 0  ; 
}


