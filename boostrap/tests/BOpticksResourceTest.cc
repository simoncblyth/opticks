#include "BOpticksResource.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ; 

    BOpticksResource res ; 

    res.Summary();

    return 0 ; 
}
