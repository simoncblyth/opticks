#include <cassert>
#include "SArgs.hh"

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 

    const char* extra = "--compute --nopropagate" ;

    SArgs sa(argc, argv, extra );

    sa.dump();

    assert(sa.hasArg("--compute"));
    assert(sa.hasArg("--nopropagate"));



    return 0 ; 
}
