#include "regexsearch.hh"
#include "assert.h"

int main(int argc, char** argv)
{
    assert(argc>1);
    std::cout << "extract enum pairs from file " << argv[1] << std::endl ;

    ipairs_t ipairs ; 
    enum_regexsearch( ipairs, argv[1] );

    dump(ipairs);

    return 0 ; 
}
