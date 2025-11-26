#include <iostream>
#include "SClientSimulator.h"

int main()
{
    stree* tree = nullptr ;
    SClientSimulator* client = new SClientSimulator(tree); ;
    std::cout 
          << " client:[" << client << "]"
          << " client.desc:[" << client->desc() << "]"
          << "\n" 
          ;

    return 0 ;
}
