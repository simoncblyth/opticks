#include <cassert>

#include "GVolume.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    GMatrix<float>* transform = NULL ; 
    GMesh* mesh = NULL ; 

    GVolume* volume = new GVolume(0, transform, mesh );

    assert(volume->getIndex() == 0);

    return 0 ;
}

