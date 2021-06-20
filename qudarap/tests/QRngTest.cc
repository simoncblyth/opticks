#include "SPath.hh"
#include "QRng.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv); 

    QRng rng ;   // loads and uploads curandState 

    LOG(info) << rng.desc() ; 


    return 0 ; 
}

