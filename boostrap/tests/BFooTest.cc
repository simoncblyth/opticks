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

    BBar bb ;
    bb.foo(d);
    bb.foo(i);
    bb.foo(n);


    BCar bc ;
    bc.foo(d);
    bc.foo(i);
    bc.foo(n);





    return 0 ; 
}
