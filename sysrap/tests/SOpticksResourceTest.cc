#include "OPTICKS_LOG.hh"

//#include "SOpticksKey.hh"
#include "SOpticksResource.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //SOpticksKey::SetKey(); 
    LOG(info) << SOpticksResource::Dump() ; 
    return 0 ; 
}
