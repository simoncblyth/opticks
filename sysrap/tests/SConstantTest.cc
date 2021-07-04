#include "OPTICKS_LOG.hh"
#include "SConstant.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info) << "SConstant::ORIGINAL_DOMAIN_SUFFIX " << SConstant::ORIGINAL_DOMAIN_SUFFIX ; 
    return 0 ; 
}
