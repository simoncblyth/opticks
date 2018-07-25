#include "X4ThreeVector.hh"

#include "OPTICKS_LOG.hh"

#include "GLMFormat.hpp"
#include "NGLMExt.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    G4ThreeVector v(1.1,2.2,3.000003 ); 

    std::string c = X4ThreeVector::Code( v, "v" ); 
    LOG(info) << c ; 

    std::string c2 = X4ThreeVector::Code( v, NULL ); 
    LOG(info) << c2 ; 


    return 0 ; 
}

