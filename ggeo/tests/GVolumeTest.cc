#include <cassert>

#include "GVolume.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    GMatrix<float>* transform = NULL ; 
    GMesh* mesh = NULL ; 
    NSensor* sensor = NULL ; 

    GVolume* volume = new GVolume(0, transform, mesh, 0, sensor );

    assert(volume->getIndex() == 0);



    return 0 ;
}

