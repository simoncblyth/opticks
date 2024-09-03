// ~/o/sysrap/tests/squad_test.sh 

#include "scuda.h"
#include "squad.h"

void test_load_array()
{
    std::array<double, 16> a ; 
    for(int i=0 ; i < 16 ; i++) a[i] = double(i*100) ; 

    quad4 p ; 
    p.load(a) ; 
    std::cout << p << std::endl ; 

}

void test_load_data()
{
    std::array<double, 16> a ; 
    for(int i=0 ; i < 16 ; i++) a[i] = double(i*100) ; 
    const double* ptr = a.data() ; 

    quad4 p ; 
    p.load(ptr, 16) ; 
    std::cout << p << std::endl ; 
}

/*
void test_dquad4()
{
    dquad4 dq4 ; 
    dq4.zero(); 

    assert( sizeof(dq4) == sizeof(double)*16 ); 

    std::cout << "dq4.d0.f " << dq4.q0.f  << std::endl ; 

    std::cout << "dq4      " << dq4 << std::endl ; 

    std::cout << "dq4.q0   " << dq4.q0 << std::endl ; 
}
*/


void test_qvals_uint3()
{

    uint3 pidxyz ; 
    qvals(pidxyz, "PIDXYZ", "-1:-1:-1" ) ; 

    std::cout << " pidxyz" << pidxyz << "\n" ; 



}


int main()
{
    /*
    test_load_array(); 
    test_load_data(); 
    test_dquad4(); 
    */
    test_qvals_uint3(); 

    return 0 ; 
}
