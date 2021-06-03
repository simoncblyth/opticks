
#include "OPTICKS_LOG.hh"
#include "CKM.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CKM ckm ; 
    ckm.init(); 
    ckm.beamOn(3); 

    return 0 ; 
}

