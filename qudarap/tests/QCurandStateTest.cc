
#include "OPTICKS_LOG.hh"
#include "QCurandState.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    unsigned long long num = 100 ;  
    unsigned long long seed = 0 ;  
    unsigned long long offset = 0 ;  

    QCurandState cs(num,seed,offset);   ; 

    LOG(info) << cs.desc() ; 

    return 0 ; 
}
