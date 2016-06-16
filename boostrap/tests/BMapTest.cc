#include "BMap.hh"

#include <map>
#include <string>


int main(int argc, char** argv)
{
    typedef std::map<std::string, std::string> SS ; 

    SS m ;
    m["hello"] = "world" ; 
    m["world"] = "hello" ; 

    BMap<std::string, std::string> bss(&m) ;
    bss.save("/tmp", "BMapTest.json") ;

    return 0 ; 
}
