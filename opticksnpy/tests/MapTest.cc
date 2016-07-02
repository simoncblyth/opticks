#include <iostream>
#include <iomanip>
#include <string>

#include "Map.hpp"

void testSaveLoad()
{
    Map<std::string, unsigned int>* a = new Map<std::string, unsigned int>() ; 

    a->add("red",1); 
    a->add("green",2); 
    a->add("blue",3); 

    const char* name = "MapTest.json" ;

    a->save("/tmp", name);

    Map<std::string, unsigned int>* b = Map<std::string, unsigned int>::load("/tmp", name);
    b->dump();

}



int main(int, char**, char** envp)
{
    typedef Map<std::string, std::string> MSS ; 
    MSS* m = new MSS();

    const char* prefix = "G4" ;

    while(*envp)
    {
       std::string kv = *envp++ ; 
       
       const size_t pos = kv.find("=") ;
       if(pos == std::string::npos) continue ;     

       //std::cout << kv << std::endl ;     

       std::string k = kv.substr(0, pos);
       std::string v = kv.substr(pos+1);
 
       std::cout << " k " << std::setw(30) << k 
                 << " v " <<std::setw(100) << v 
                 << std::endl ;  

       if(k.find(prefix)==0) m->add(k,v);  // startswith
    }

    m->save("/tmp", "G4.ini");


    return 0 ; 
}
