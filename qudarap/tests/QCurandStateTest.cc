
#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "QCurandState.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    unsigned long long num = SSys::getenvint("NUM", 1 ) ;  
    if(num < 20) num *= 1000000 ; // num less than 20 assumed to be in millions  

    unsigned long long seed = 0 ;  
    unsigned long long offset = 0 ;  

    QCurandState cs(num,seed,offset);   ; 

    LOG(info) << cs.desc() ;

    return 0 ; 
}
