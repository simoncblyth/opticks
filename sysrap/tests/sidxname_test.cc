// ./sidxname_test.sh

#include "sidxname.h"
#include <iostream>
#include <set>
#include <vector>

int main(int argc, char** argv)
{
    sidxname a(0, "Red") ; 
    sidxname b(1, "Green") ; 
    sidxname c0(2, "Blue") ; 
    sidxname c1(2, "Blue") ; 
    //sidxname d(-1, "0123456789abcdef0123456789abcdef") ; 
    //sidxname e(-2, "0123456789abcdef0123456789abcdef_") ; 

    sidxname d(-1, "0123456789abcdef0123456789abcde") ; 
    sidxname e(-2, "0123456789abcdef0123456789abcde") ; // maximum is 31

    std::set<sidxname,sidxname_ordering> mm ; 
    mm.insert(a);  
    mm.insert(b);  
    mm.insert(c0);  
    mm.insert(c1);  
    mm.insert(d); 
    mm.insert(e); 

    std::vector<sidxname> vmm(mm.begin(), mm.end()) ; 
    for(int i=0 ; i < vmm.size() ; i++) std::cout << vmm[i].desc() << std::endl ; 

    return 0 ; 
}
