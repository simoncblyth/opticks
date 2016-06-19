#include <string>
#include "Map.hpp"

int main()
{
    Map<std::string, unsigned int>* a = new Map<std::string, unsigned int>() ; 

    a->add("red",1); 
    a->add("green",2); 
    a->add("blue",3); 

    const char* name = "MapTest.json" ;

    a->save("/tmp", name);


    Map<std::string, unsigned int>* b = Map<std::string, unsigned int>::load("/tmp", name);
    b->dump();


    return 0 ; 
}
