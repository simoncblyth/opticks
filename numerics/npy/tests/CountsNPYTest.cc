#include "CountsNPY.hpp"
#include <map>

int main()
{

    std::map<std::string, int> m ;
    m["red"] = 1 ; 
    m["green"] = 2 ; 
    m["blue"] = 3 ; 

    CountsNPY<int> c("hello");
    c.add(m);
    c.sort();
    c.dump();

    c.sort(false);
    c.dump();




}
