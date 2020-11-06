#pragma once
#include <cstddef>
#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SRand
{
    public:
        static unsigned pick_random_category(unsigned num_cat);  // randomly returns value : 0,..,num_cat-1
};

