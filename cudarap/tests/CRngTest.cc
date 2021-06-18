#include "CRng.hh"
#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv); 

    const char* path = "/Users/blyth/.opticks/rngcache/RNG/cuRANDWrapper_1000000_0_0.bin" ; 
    CRng rng(path);   // loads and uploads to GPU 


    return 0 ; 
}

