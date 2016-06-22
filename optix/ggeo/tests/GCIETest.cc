

#include "NGLM.hpp"
#include "GCIE.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ; 


    GCIE cie(380.,780,40.);
    cie.dump();

    return 0 ; 
}
