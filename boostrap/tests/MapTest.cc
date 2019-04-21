// TEST=MapTest om-t 

#include <iostream>
#include <iomanip>
#include <string>

#include "OPTICKS_LOG.hh"
#include "Map.hh"

void testSaveLoad()
{
    Map<std::string, unsigned int>* a = new Map<std::string, unsigned int>() ; 

    a->add("red",1); 
    a->add("green",2); 
    a->add("blue",3); 

    const char* name = "MapTest.json" ;

    a->save("$TMP", name);

    Map<std::string, unsigned int>* b = Map<std::string, unsigned int>::load("$TMP", name);
    b->dump();

}


void testEnvSpin( char** envp )
{

    typedef Map<std::string, std::string> MSS ; 
    MSS* m = new MSS();

    const char* prefix = "G4" ;

    while(*envp)
    {
       std::string kv = *envp++ ; 
       
       const size_t pos = kv.find("=") ;
       if(pos == std::string::npos) continue ;     

       std::string k = kv.substr(0, pos);
       std::string v = kv.substr(pos+1);
 
       std::cout 
           << " k " << std::setw(30) << k 
           << " v " <<std::setw(100) << v 
           << std::endl ;  

       if(k.find(prefix)==0) m->add(k,v);  // startswith
    }

    m->save("$TMP", "G4.ini");
}



void testGet()
{
    typedef Map<std::string, std::string> MSS ; 
    MSS* a = new MSS ; 

    a->add("red","a"); 
    a->add("green","b"); 
    a->add("blue","c"); 

    assert( a->hasKey("red") == true ) ; 
    assert( a->hasKey("cyan") == false ) ; 
}



int main(int argc, char** argv, char** envp)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info); 

    //testEnvSpin(envp); 
    testGet();


    return 0 ; 
}
