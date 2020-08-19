#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "ImageNPY.hpp"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* path = argc > 1 ? argv[1] : "/tmp/SPPMTest.ppm" ; 
    const bool yflip = false ; 
    const unsigned ncomp = 3 ; 

    NPY<unsigned char>* a = ImageNPY::LoadPPM(path, yflip, ncomp ) ; 
    //a->dump();   // dumping of unsigned char array gives mess 

    a->save("$TMP/ImageNPYTest.npy"); 

    return 0 ; 
}

