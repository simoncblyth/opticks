// ./squad_test.sh 

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




int main()
{
    /*
    test_load_array(); 
    */
    test_load_data(); 

    return 0 ; 
}
