#include "OPTICKS_LOG.hh"

int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv) ; 

    OPTICKS_LOG_::Check();

    LOG(info) << argv[0] ; 

    return 0  ; 
}


