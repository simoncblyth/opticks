#include "regexsearch.hh"
#include "assert.h"

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "$ENV_HOME/graphics/ggeoview/cu/photon.h" ;

    std::cout << "extract enum pairs from file " << path << std::endl ;

    upairs_t upairs ; 
    enum_regexsearch( upairs, path );

    udump(upairs);

    return 0 ; 
}
