// name=SComp_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <cassert>
#include <iostream>
#include "SComp.h"

void test_Mask()
{
    const char* names = "photon,record,rec" ; 
    unsigned mask = SComp::Mask(names); 

    std::cout 
        << " names " << names << std::endl  
        << " SComp::Mask(names) " << SComp::Mask(names) << std::endl 
        << " mask " << mask << std::endl 
        << " SComp::Desc(mask) " << SComp::Desc(mask) << std::endl 
        ; 
}

void test_CompListMask()
{
    const char* names = "photon,record,rec" ; 
    unsigned mask = SComp::Mask(names) ; 
    std::vector<unsigned> comps ; 
    SComp::CompListMask(comps, mask ); 

    std::string desc = SComp::Desc(comps) ; 

    std::cout 
        << " test_CompListMask "
        << " names " << names
        << " mask " << std::hex << mask << std::dec
        << " comps.size " << comps.size() 
        << " SComp::Desc(comps) " << desc 
        << std::endl 
        ; 

    assert( strcmp( names, desc.c_str()) == 0 ); 

}






int main()
{
    //test_Mask(); 
    test_CompListMask(); 

    return 0 ; 
}
