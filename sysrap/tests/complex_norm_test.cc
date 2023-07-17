// name=complex_norm_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

/**

https://en.cppreference.com/w/cpp/numeric/complex/norm

**/


#include <cassert>
#include <complex>
#include <iostream>
 
int main()
{
    std::complex<double> z {3.0, 4.0};

    assert(std::norm(z) == (z.real() * z.real() + z.imag() * z.imag()));
    assert(std::norm(z) == (z * std::conj(z)));
    assert(std::norm(z) == (std::abs(z) * std::abs(z)));

    std::cout << "std::norm(" << z << ") = " << std::norm(z) << '\n';


    for(double v=2.0 ; v > -2.1 ; v-=0.1 )
    {
        std::complex<double> q {v, 0.0 } ; 
        std::cout << "std::norm(" << q << ") = " << std::norm(q) << '\n';
    }

    return 0 ; 
}

