// ./scuda_double_test.sh 

#include "scuda.h"
#include "scuda_double.h"

void test_length()
{
    double2 d2 = { 3., 4. } ; 
    double l_d2 = length(d2) ; 
    std::cout << " d2 " << d2 << " l_d2 " << l_d2 << std::endl ; 

    double s = 1./sqrtf(3.) ; 
    double3 d3 = { s, s, s } ; 
    double l_d3 = length(d3) ; 
    std::cout << " d3 " << d3 << " l_d3 " << l_d3 << std::endl ; 
}


int main()
{
    test_length(); 

    return 0 ; 
}
