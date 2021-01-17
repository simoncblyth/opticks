#include "X4Named.hh"
#include "X4NameOrder.hh"
#include "OPTICKS_LOG.hh"

template struct X4NameOrder<X4Named> ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);  

    std::vector<X4Named*> a ; 
    a.push_back(new X4Named("red0x001")); 
    a.push_back(new X4Named("green0x002")); 
    a.push_back(new X4Named("blue0x003")); 
    a.push_back(new X4Named("cyan0x003")); 
    a.push_back(new X4Named("magenta0x003")); 
    a.push_back(new X4Named("yellow0x003")); 

    X4NameOrder<X4Named>::Dump("asis", a ); 

    bool reverse = false ; 
    bool strip = true ; 
    X4NameOrder<X4Named> order(reverse, strip); 
    std::sort( a.begin(), a.end(), order ); 

    X4NameOrder<X4Named>::Dump("after sort", a ); 

    return 0 ; 
} 


