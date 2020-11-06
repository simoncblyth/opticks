#include <cstdlib>
#include "SRand.hh"


/**
SRand::pick_random_category
----------------------------

Randomly returns value : 0,..,num_cat-1

**/

unsigned SRand::pick_random_category(unsigned num_cat)  
{
    unsigned u = rand() ; 
    return u % num_cat ; 
}
