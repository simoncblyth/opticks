#include "OPTICKS_LOG.hh"

#include "SLog.hh"

int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv) ; 

    OPTICKS_LOG_::Check();

    SLog::Nonce();

    LOG(info) << argv[0] ; 

    return 0  ; 
}


