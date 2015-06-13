#include "regexsearch.hh"
#include "assert.h"

int main(int argc, char** argv)
{
    assert(argc>1);
    std::cout << "extract enum pairs from file " << argv[1] << std::endl ;

    upairs_t upairs ; 
    enum_regexsearch( upairs, argv[1] );

    udump(upairs);

    return 0 ; 
}
