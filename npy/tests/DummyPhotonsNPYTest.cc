#include "SSys.hh"
#include "OPTICKS_LOG.hh"

#include "DummyPhotonsNPY.hpp"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned hitmask = 0x1 << 6 ;  // 64
    NPY<float>* npy = DummyPhotonsNPY::Make(100, hitmask);
    const char* path = "$TMP/DummyPhotonsNPYTest.npy" ;
    npy->save(path);

    SSys::npdump(path);

    return 0 ; 
}
