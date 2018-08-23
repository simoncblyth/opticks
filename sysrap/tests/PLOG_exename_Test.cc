#include "OPTICKS_LOG.hh"
#include "SLog.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* exename = PLOG::instance->args.exename() ;    

    LOG(info) << " exename " << exename ; 

    const char* exename2 = SLog::exename() ;  

    LOG(info) << " exename2 " << exename2 ; 



    return 0 ; 
}
