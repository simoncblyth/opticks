#include "TorchStepNPY.hpp"

#include "SSys.hh"
#include "NPY.hpp"

#ifdef _MSC_VER
// quell: object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif

#include "SYSRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "PLOG.hh"




int main(int argc, char** argv )
{

    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 
    NPY_LOG__ ; 


    TorchStepNPY* m_torchstep ; 


    unsigned TORCH = 4096 ; 
    unsigned nstep = 1 ; 
    //const char* config = "target=3153;photons=10000;dir=0,1,0" ;
    const char* config = NULL ;

    m_torchstep = new TorchStepNPY(TORCH, nstep, config);

    m_torchstep->dump();

    m_torchstep->addStep();

    NPY<float>* npy = m_torchstep->getNPY();
    npy->save("$TMP/torchstep.npy");

    assert(npy->getNumItems() == nstep);

    const char* cmd = "python -c 'import os, numpy as np ; print np.load(os.path.expandvars(\"$TMP/torchstep.npy\")).view(np.int32)' " ;
    system(cmd);

    SSys::npdump("$TMP/torchstep.npy", "np.int32");
    SSys::npdump("$TMP/torchstep.npy", "np.float32");


    return 0 ;
}


