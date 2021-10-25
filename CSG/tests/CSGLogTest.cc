#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{ 
    OPTICKS_LOG_::Check(); 

    OPTICKS_LOG(argc, argv); 

    OPTICKS_LOG_::Check(); 

    LOG(info) ; 

    return 0 ; 
}
