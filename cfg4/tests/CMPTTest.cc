#include "OPTICKS_LOG.hh"
#include "CMPT.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    CMPT* mpt = CMPT::MakeDummy() ; 
    mpt->dump(); 

    return 0 ; 
}

