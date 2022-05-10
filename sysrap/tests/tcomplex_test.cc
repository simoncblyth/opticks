// name=tcomplex_test ; gcc $name.cc -g -std=c++11 -I.. -I/usr/local/cuda/include  -lstdc++ -o /tmp/$name && /tmp/$name

#include "tcomplex.h"

int main()
{
    cuFloatComplex a = tcomplex::make_cuFloatComplex_polar( 0.f, 0.f ); 
    cuFloatComplex b = tcomplex::cuSqrtf(a); 

    std::cout << " a " << a << " b " << b << std::endl ; 

    return 0 ; 
}

