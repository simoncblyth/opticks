#include "OPTICKS_LOG.hh"

#include "SLog.hh"

int main(int argc , char** argv )
{
    OPTICKS_LOG__(argc, argv) ; 
    //OPTICKS_LOG_COLOR__(argc, argv) ; 

    SLog::Nonce();

    LOG(info) << argv[0] ; 

    return 0  ; 
}


