
#include "OXPPNS.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    optix::Context context = optix::Context::create(); 

    return 0 ; 
}


