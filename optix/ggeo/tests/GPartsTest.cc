
#include <cassert>
#include "GParts.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    GParts* pts = new GParts ; 
    assert(pts->getNumParts() == 0 );
 

    return 0 ;
}

