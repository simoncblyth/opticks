/**
SOpticksResourceTest.cc
=========================

The output from this executable is used by the bash function : opticks-key-remote-dir::

    opticks-key-remote-dir () 
    { 
        local msg="=== $FUNCNAME ";
        [ -z "$OPTICKS_KEY_REMOTE" ] && echo $msg missing required envvar OPTICKS_KEY_REMOTE && return 1;
        local opticks_key_remote_dir=$(OPTICKS_KEY=$(opticks-key-remote) OPTICKS_GEOCACHE_PREFIX=.opticks SOpticksResourceTest --keydir);
        echo $opticks_key_remote_dir
    }

Thus avoid emitting anything extra when the --cfbase --key  and --keydir arguments are used.  

**/

#include "OPTICKS_LOG.hh"
#include "SOpticksResource.hh"
#include "SOpticksKey.hh"
#include "SStr.hh"



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
