#include "GenstepNPY.hpp"

#include "SSys.hh"
#include "NPY.hpp"

#include "SYSRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "PLOG.hh"


unsigned TORCH = 4096 ; 


void test_fabstep_0()
{
    unsigned nstep = 10 ; 
    GenstepNPY* fab = new GenstepNPY(TORCH, nstep) ;  

    for(unsigned i=0 ; i < nstep ; i++)
    {
        fab->setMaterialLine(i*10);
        fab->setNumPhotons(1000); 
        fab->addStep();
    }
    NPY<float>* npy = fab->getNPY();

    const char* path = "$TMP/fabstep_0.npy" ;
    npy->save(path);

    SSys::npdump(path, "np.int32");
}

void test_fabstep_1()
{
    unsigned nstep = 10 ; 
    GenstepNPY* fab = GenstepNPY::Fabricate(TORCH, nstep) ;  

    NPY<float>* npy = fab->getNPY();
    const char* path = "$TMP/fabstep_1.npy" ;
    npy->save(path);
    SSys::npdump(path, "np.int32");
}


int main(int argc, char** argv )
{

    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 
    NPY_LOG__ ; 

    test_fabstep_0();
    test_fabstep_1();


    return 0 ;
}


