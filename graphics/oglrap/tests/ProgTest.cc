#include <cstdlib>
#include "Prog.hh"


#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv); 

    Prog prog(getenv("SHADER_DIR"), "nrm");
    return 0 ;
}


