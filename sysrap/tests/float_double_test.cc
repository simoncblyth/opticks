// D=1.00001 ~/opticks/sysrap/tests/float_double_test.sh 

#include "ssys.h"
#include <iostream>
#include <iomanip>

int main()
{
    double d = ssys::getenvdouble("D", 1.00000100 ) ; 
    float  f = d ; 

    std::cout << " d " << std::fixed << std::setw(10) << std::setprecision(8) << d << std::endl ; 
    std::cout << " f " << std::fixed << std::setw(10) << std::setprecision(8) << f << std::endl ; 

    return 0 ; 

}
