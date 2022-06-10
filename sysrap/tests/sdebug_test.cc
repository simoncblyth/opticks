// name=sdebug_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include "sstr.h"
#include "sdebug.h"

int main()
{
    sdebug* dbg = new sdebug ; 
    dbg->zero(); 
    dbg->d12match_fail = 10 ; 
    std::cout << dbg->desc() << std::endl ; 
    return 0 ; 
}

