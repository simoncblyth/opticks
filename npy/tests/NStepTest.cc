#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "NPY.hpp"
#include "NStep.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    NStep onestep ; 
    onestep.setGenstepType(42); 
    onestep.setNumPhotons(101); 
    onestep.fillArray(); 

    LOG(info) << onestep.desc() ; 

    NPY<float>* ary = onestep.getArray(); 

    const char* path = "$TMP/NStepTest.npy" ; 
    ary->save(path); 

    SSys::npdump(path, "np.float32"); 
    SSys::npdump(path, "np.uint32"); 

    return 0 ; 
}
