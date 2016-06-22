#include <cassert>

#include "GSolid.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    GMatrix<float>* transform = NULL ; 
    GMesh* mesh = NULL ; 
    NSensor* sensor = NULL ; 

    GSolid* solid = new GSolid(0, transform, mesh, 0, sensor );

    assert(solid->getIndex() == 0);



    return 0 ;
}

