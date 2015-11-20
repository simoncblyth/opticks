#include "NPart.hpp"
#include "GLMPrint.hpp"


void npart::dump(const char* msg)
{
    print(q0.f, "q0.f");
    print(q1.f, "q1.f");
    print(q2.f, "q2.f");
    print(q3.f, "q3.f");

    print_u(q0.u, "q0.u");
    print_u(q1.u, "q1.u");
    print_u(q2.u, "q2.u");
    print_u(q3.u, "q3.u");

    print_i(q0.i, "q0.i");
    print_i(q1.i, "q1.i");
    print_i(q2.i, "q2.i");
    print_i(q3.i, "q3.i");


}



