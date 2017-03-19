#include <cstring>
#include "NBBox.hpp"

void nbbox::dump(const char* msg)
{
    printf("%s\n", msg);
    min.dump("bb min");
    max.dump("bb max");
}

void nbbox::include(const nbbox& other)
{
    min = nminf( min, other.min );
    max = nmaxf( max, other.max );
}


