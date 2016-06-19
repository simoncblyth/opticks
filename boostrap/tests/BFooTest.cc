#include "BFoo.hh"

int main(int, char** argv)
{
    std::cerr << argv[0] << std::endl ; 

    double d = 10. ; 
    int i = 42 ; 
    char* n = argv[0] ;


    foo(d);
    foo(i);
    foo(n);

    return 0 ; 
}
