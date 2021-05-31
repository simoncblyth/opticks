#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "NPY.hpp"
#include "NStep.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    NStep one ; 
    one.setGenstepType(42); 
    one.setNumPhotons(101); 
    one.fillArray(); 

    LOG(info) << one.desc() ; 

    NPY<float>* ary = one.getArray(); 

    const char* path = "$TMP/NStepTest.npy" ; 
    ary->save(path); 

    SSys::npdump(path, "np.float32"); 
    SSys::npdump(path, "np.uint32"); 

    return 0 ; 
}
