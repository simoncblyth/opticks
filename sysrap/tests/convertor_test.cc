// name=convertor_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>

struct TCheck
{
    double fRe ; 
    double fIm ; 

    // TComplex has a convertor like this : looks like a bug invitation to me 
    operator double () const { return fRe; }
};


int main()
{
    TCheck c {1., 100. } ; 

    double c0 = c ; 
    std::cout << " c0 " << c0 << std::endl ; 

    return 0 ; 
}
