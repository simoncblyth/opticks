#include "OPTICKS_LOG.hh"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    NPY<unsigned>* a = NPY<unsigned>::load("/tmp/X4PhysicalVolume/extras/2/indices.npy");

    a->dump();

    a->reshape(-1,1); 

    a->dump();

    return 0 ; 
}
