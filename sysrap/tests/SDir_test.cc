// export OPTICKS_RANDOM_SEQPATH=/tmp/$USER/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000
// name=SDir_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include <cstdlib>
#include "SDir.h"

int main()
{
    std::vector<std::string> names ; 
    SDir::List(names, getenv("OPTICKS_RANDOM_SEQPATH"), ".npy" );
    std::cout << SDir::Desc(names) << std::endl ;

    return 0 ; 
}
