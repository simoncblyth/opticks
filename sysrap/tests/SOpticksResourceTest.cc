/**
SOpticksResourceTest.cc : THIS IS GOING THE WAY OF THE DODO 
==============================================================

**/

#include "OPTICKS_LOG.hh"
#include "SOpticksResource.hh"
#include "SPath.hh"



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* arg = argc > 1 ? argv[1] : nullptr ; 

    if( arg == nullptr )
    {
        LOG(info) << SOpticksResource::Dump() ; 
        std::cout << SOpticksResource::Desc() << std::endl ; 
    }
    else if( strcmp(arg, "--cfbase") == 0 )
    {
        std::cout << SOpticksResource::CFBase() << std::endl ; 
    }
    else if( strcmp(arg, "--resolve") == 0 )
    {
        const char* path = SPath::Resolve("$PrecookedDir", NOOP) ; 
        std::cout << "SPath::Resolve(\"$PrecookedDir\", NOOP) " << path  << std::endl ; 
    }
    else if( strcmp(arg, "--dod") == 0 )
    {
        const char* path = SPath::Resolve("$DefaultOutputDir", NOOP) ; 
        std::cout << "SPath::Resolve(\"$DefaultOutputDir\", NOOP) " << path  << std::endl ; 
    }
    else if( strcmp(arg, "--ddod") == 0 )
    {
        std::string desc = SOpticksResource::Desc_DefaultOutputDir() ; 
        std::cout << "SOpticksResource::Desc_DefaultOutputDir() " << std::endl << desc  << std::endl ; 
    }
    else if( strcmp(arg, "--exe") == 0 )
    {
        const char* exe = SOpticksResource::ExecutableName() ; 
        std::cout << "SOpticksResource::ExecutableName() " << exe  << std::endl ; 
    }
    else
    {
        LOG(error) << " arg [" << arg << "] is not handled " ; 
    }

 
    return 0 ; 
}
// om- ; TEST=SOpticksResourceTest om-t


