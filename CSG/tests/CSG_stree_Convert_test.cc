// ./CSG_stree_Convert_test.sh  

#include <iostream>
#include "CSG_stree_Convert.h" 

int main()
{
    const stree* st = stree::Load() ; 
    if(!st) return 1 ; 
    std::cout << st->desc() ; 

    CSGFoundry* fd = CSG_stree_Convert::Translate(st) ; 
    std::cout << fd->desc() ; 

    return 0 ; 
}

