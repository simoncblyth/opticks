#include <iostream>
#include "s_csg.h"

int main()
{
    s_csg::Load("$BASE");  
    std::cerr << s_csg::Desc() ;  

    return 0 ; 
}
