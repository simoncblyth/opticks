#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    OPTICKS_LOG_::Check();
    return 0 ; 
}
