// ./stmm_test.sh

#include <cstdlib>
#include "stmm.h"

#include "scuda.h"
#include "squad.h"
#include "sphoton.h"


typedef double T ; 
//typedef float T ; 


void test_StackSpec_eget()
{
    putenv((char*)"L0=1,0,0");   
    putenv((char*)"L1=1.5,0,0");   

    StackSpec<T,2> ss ; 
    ss.eget();  

    std::cout << ss ; 
}


void test_StackSpec_Create2()
{
    StackSpec<T,2> ss(StackSpec<T,2>::Create2(1.0, 1.5)) ; 
    std::cout << ss ; 
}





int main(int argc, char** argv)
{
    T mct = argc > 1 ? std::atof(argv[1]) : -1.f  ;   // minus_cos_theta
    T wl  = argc > 2 ? std::atof(argv[2]) : 440.f ;   // wavelength_nm

    /*
    test_StackSpec_eget(); 
    */
    test_StackSpec_Create2(); 


    /*
    Stack<T,2> stack(wl, mct, ss ); // ART calc done in ctor    

    std::cout << ss << stack ; 

    */



    return 0 ; 
} 
