// ./sproplist_test.sh 

#include <iostream>
#include <iomanip>
#include "sproplist.h"

void test_sprop_parse(const char* str)
{
    sprop p ; 
    std::cout << ( p.parse(str) ? p.desc() : "FAIL" ) << std::endl ; 
}

void test_sprop_parse()
{
    const char* str0 = "   2 3 RINDEX 1.0  " ; 
    const char* str1 = "   2 3 " ; 
    test_sprop_parse(str0); 
    test_sprop_parse(str1); 
}

void test_sproplist_Material()
{
    const sproplist* pm = sproplist::Material() ; 
    std::cout << "pm.desc" << std::endl << pm->desc() ; 

    std::vector<std::string> names ; 
    pm->getNames(names) ; 

    std::cout << "names.size " << names.size() << std::endl ; 
    for(size_t i=0 ; i < names.size() ; i++) 
    {
        const char* name = names[i].c_str(); 
        const sprop* prop = pm->findProp(name) ; 
        std::cout << std::setw(20) << name << " : " << prop->desc() << std::endl ; 
    }

    for(int g=0 ; g < sprop::NUM_PAYLOAD_GRP ; g++)
    {
        for(int v=0 ; v < sprop::NUM_PAYLOAD_VAL ; v++)
        {
            const sprop* prop = pm->get(g,v) ; 
            std::cout << " pm.get(" << g << "," << v << ") " << prop->desc() << std::endl ; 
        }
    }
}


int main(int argc, char** argv)
{
    test_sproplist_Material(); 

    return 0 ; 
}
