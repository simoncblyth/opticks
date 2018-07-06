#include <cstdlib>
#include "Prog.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Prog prog(getenv("SHADER_DIR"), "nrm");
    return 0 ;
}


