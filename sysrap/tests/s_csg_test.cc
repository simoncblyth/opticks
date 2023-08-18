#include <iostream>
#include "s_csg.h"

int main()
{
    s_csg::Load("$BASE");  
    std::cerr << s_csg::Desc() ;  

    sn* nd = s_csg::INSTANCE->get_nd(0) ; 

    std::cerr 
        << " nd " << ( nd ? nd->desc() : "-" )
        << std::endl 
        ;


    return 0 ; 
}
