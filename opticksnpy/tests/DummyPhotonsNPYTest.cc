#include "SSys.hh"
#include "PLOG.hh"

#include "DummyPhotonsNPY.hpp"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NPY<float>* npy = DummyPhotonsNPY::make(100);
    const char* path = "$TMP/DummyPhotonsNPYTest.npy" ;
    npy->save(path);

    SSys::npdump(path);

    return 0 ; 
}
