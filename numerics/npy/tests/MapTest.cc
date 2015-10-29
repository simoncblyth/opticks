#include "Map.hpp"

int main()
{
    Map* a = new Map ; 

    a->add("red",1); 
    a->add("green",2); 
    a->add("blue",3); 

    a->save("/tmp", "test.json");


    Map* b = Map::load("/tmp", "test.json");
    b->dump();


    return 0 ; 
}
