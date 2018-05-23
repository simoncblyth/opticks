#include "FabStepNPY.hpp"

#include "SSys.hh"
#include "NPY.hpp"

#include "SYSRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "PLOG.hh"


unsigned TORCH     =  0x1 << 12 ; 
unsigned FABRICATED = 0x1 << 15 ; 


void test_fabstep_0()
{
    unsigned nstep = 10 ; 
    FabStepNPY* fab = new FabStepNPY(FABRICATED, nstep, 100 ) ;  
    NPY<float>* npy = fab->getNPY();

    const char* path = "$TMP/fabstep_0.npy" ;
    npy->save(path);

    SSys::npdump(path, "np.int32");
}

int main(int argc, char** argv )
{

    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 
    NPY_LOG__ ; 

    test_fabstep_0();


    return 0 ;
}


