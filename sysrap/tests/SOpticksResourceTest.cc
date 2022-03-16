#include "OPTICKS_LOG.hh"
#include "SOpticksResource.hh"
#include "SOpticksKey.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* arg = argc > 1 ? argv[1] : nullptr ; 

    if( arg == nullptr )
    {
        LOG(info) << SOpticksResource::Dump() ; 
    }
    else if( strcmp(arg, "--cfbase") == 0 )
    {
        std::cout << SOpticksResource::CFBase() << std::endl ; 
    }
    else if( strcmp(arg, "--key") == 0 )
    {
        std::cout << SOpticksKey::Key() << std::endl ; 
    }
    else if( strcmp(arg, "--keydir") == 0 )
    {
        std::cout << SOpticksResource::IDPath(true) << std::endl ; 
    }
    else
    {
        LOG(error) << " arg [" << arg << "] is not handled " ; 
    }
    return 0 ; 
}
