/**
S4MultiLayrStack_test.cc
=========================

~/o/sysrap/tests/S4MultiLayrStack_test.sh


**/

#include "S4MultiLayrStack.h"
#include "ssys.h"

struct S4MultiLayrStack_test
{
    template<typename F>
    static int demo();

    static int main();
};


template<typename F>
int S4MultiLayrStack_test::demo()
{
    F ss[16] = 
      {   
        1.482,     0,      0,   0,  
        1.920,     0,  36.49,   0,  
        2.429, 1.366,  21.13,   0,  
        1.000,     0,      0,   0   
      };  

    F wl = 440. ; 
    F minus_cos_theta = -1. ; 
    F dot_pol_cross_mom_nrm = 0. ; 
    
    Stack<F,4> stack  ;   
    stack.calc( wl, minus_cos_theta, dot_pol_cross_mom_nrm, ss, 16u );  

    //std::cout << stack ; 
    std::cout << stack.art ; 

    return 0 ;
}

int S4MultiLayrStack_test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;
    if(ALL|| 0 == strcmp(TEST,"demof")) rc += demo<float>();
    if(ALL|| 0 == strcmp(TEST,"demod")) rc += demo<double>();

    return rc ;

}

int main(){ return S4MultiLayrStack_test::main(); }

