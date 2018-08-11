//#include "SProc.hh"
#include "SSys.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "Opticks.hh"
#include "OpticksHub.hh"

#include "CCollector.hh"

#include "OPTICKS_LOG.hh"


unsigned mock_num_steps( unsigned e )
{
    unsigned ns = 0 ; 
    switch( e % 10 )
    {
       case 0: ns = 10 ; break ;
       case 1: ns = 50 ; break ;
       case 2: ns = 60 ; break ;
       case 3: ns = 80 ; break ;
       case 4: ns = 10 ; break ;
       case 5: ns = 100 ; break ;
       case 6: ns = 30 ; break ;
       case 7: ns = 300 ; break ;
       case 8: ns = 20 ; break ;
       case 9: ns = 10 ; break ;
   }
   return ns ;  
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned mock_num_evt = 100 ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);

    CCollector cc(hub.getLookup());
    NPY<float>* gs = cc.getGensteps(); 


    for(unsigned e=0 ; e < mock_num_evt ; e++)
    {
        unsigned num_steps = mock_num_steps(e); 

        for(unsigned i=0 ; i < num_steps ; i++) CCollector::Instance()->collectMachineryStep(e*1000+i);

        const char* path = SSys::fmt("$TMP/CCollectorTest%u.npy",e) ;
        gs->save(path);
        gs->reset();

        SSys::npdump(path, "np.uint32" );

        ok.profile(e) ; 
    }

    ok.dumpProfile(argv[0]); 
    ok.saveProfile();
   

    return ok.getRC() ; 
}


/*

In [1]: import sys, os, numpy as np ; np.set_printoptions(suppress=True, precision=3)

In [2]: a=np.load(os.path.expandvars("$TMP/CCollectorTest_vm.npy"))

In [3]: a.shape
Out[3]: (100,)

In [4]: a
Out[4]: 
array([  1.,  10.,  10.,  11.,  19.,  19.,  19.,  20.,  28.,  28.,  28.,
        28.,  28.,  28.,  30.,  38.,  38.,  38.,  38.,  38.,  38.,  39.,
        47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,
        47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,
        47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,
        47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,
        47.,  47.,  47.,  48.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,
        57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,
        57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,
        57.], dtype=float32)

In [5]: import matplotlib.pyplot as plt 

In [6]: plt.plot(a)
Out[6]: [<matplotlib.lines.Line2D at 0x112f0bdd0>]

In [7]: plt.show()


*/



