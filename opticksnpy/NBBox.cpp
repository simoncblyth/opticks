//#include <sstream>
#include <cstring>
#include "NBBox.hpp"

const char* nbbox::desc() const
{
    char _desc[128];
    snprintf(_desc, 128, " mi %.32s mx %.32s ", min.desc(), max.desc() );
    return strdup(_desc);
}


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


