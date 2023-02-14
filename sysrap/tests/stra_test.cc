// name=stra_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/glm/glm -o /tmp/$name && /tmp/$name


#include <sstream>
#include <iostream>
#include <iomanip>

#include "stra.h"

int main()
{
    glm::tmat4x4<double> a(1.); 
    glm::tmat4x4<double> b(2.); 
    glm::tmat4x4<double> c(3.); 

    std::cout << stra<double>::Desc(a, b, c, "a", "b", "c" ); 

    return 0 ; 
}
