#include "NPart.hpp"
#include <cstdio>

void npart::dump(const char* msg)
{
    printf("%s\n", msg);
    q0.dump("q0");
    q1.dump("q1");
    q2.dump("q2");
    q3.dump("q3");
}



