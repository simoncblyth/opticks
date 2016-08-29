#include <cstdlib>
#include <cstdio>

#include "PLOG.hh"
#include "SYSRAP_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 

    LOG(info) << " hello from " << argv[0] ; 


    return 0 ; 
}
