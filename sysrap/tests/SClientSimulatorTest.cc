#include "SClientSimulator.h"

int main()
{
    SClientSimulator* client = SClientSimulator::Create("$CFBaseFromGEOM/CSGFoundry/SSim"); ;
    if(!client) return 1 ;
    std::cout << " client.desc:[" << client->desc() << "]\n" ;

    for(int i=0 ; i < 10 ; i++) client->simulate(i, false );

    return 0 ;
}


