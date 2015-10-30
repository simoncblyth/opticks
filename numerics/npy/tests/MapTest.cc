#include <string>
#include "Map.hpp"

int main()
{
    Map<std::string, unsigned int>* a = new Map<std::string, unsigned int>() ; 

    a->add("red",1); 
    a->add("green",2); 
    a->add("blue",3); 

    a->save("/tmp", "test.json");


    Map<std::string, unsigned int>* b = Map<std::string, unsigned int>::load("/tmp", "test.json");
    b->dump();


    return 0 ; 
}
