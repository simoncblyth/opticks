#include "SProc.hh"
#include "SSys.hh"

#include "NGLM.hpp"
#include "NPY.hpp"
#include "SYSRAP_LOG.hh"

#include "Opticks.hh"
#include "OpticksHub.hh"

#include "CCollector.hh"


#include "PLOG.hh"



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
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 


    unsigned mock_num_evt = 100 ; 

    NPY<float>* vm = NPY<float>::make(mock_num_evt) ;  
    vm->zero();
    float* vmv = vm->getValues();

    Opticks ok(argc, argv);

    OpticksHub hub(&ok);

    CCollector cc(&hub);
    NPY<float>* gs = cc.getGensteps(); 

    float vmb0 = SProc::VirtualMemoryUsageMB();

    for(unsigned e=0 ; e < mock_num_evt ; e++)
    {
        unsigned num_steps = mock_num_steps(e); 

        for(unsigned i=0 ; i < num_steps ; i++) CCollector::Instance()->collectMachineryStep(e*1000+i);

        const char* path = SSys::fmt("$TMP/CCollectorTest%u.npy",e) ;
        gs->save(path);
        gs->reset();

        SSys::npdump(path, "np.uint32" );

        float dvmb = SProc::VirtualMemoryUsageMB() - vmb0 ;
        vmv[e] = dvmb ; 
    }

    const char* vmpath = "$TMP/CCollectorTest_vm.npy" ;
    vm->save(vmpath);
    SSys::npdump(vmpath, "np.float32" );


    return ok.getRC() ; 
}
