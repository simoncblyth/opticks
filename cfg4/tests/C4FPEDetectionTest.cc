#include "OPTICKS_LOG.hh"
#include "G4FPEDetection.hh"
#include "C4FPEDetection.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //LOG(info) << 1/0 ; 

    InvalidOperationDetection(); 
    C4FPEDetection::InvalidOperationDetection_Disable(); 

    LOG(info) << 1/0 ; 
    

    return 0 ; 
}

