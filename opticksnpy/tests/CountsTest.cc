#include "Counts.hpp"
#include "Index.hpp"
#include <map>

int main()
{
    typedef unsigned int T ; 


    std::map<std::string, T> m ;
    m["red"] = 1 ; 
    m["green"] = 2 ; 
    m["blue"] = 3 ; 

    Counts<T> c("hello");
    c.addMap(m);
    c.sort();
    c.dump();

    c.sort(false);
    c.dump();


    c.checkfind("green");


    c.add("purple");
    c.add("green", 10);
    c.add("red", 2);

    c.dump();
    c.sort();
    c.dump();



    Counts<T> t("test");
    t.add("red",  1); 
    t.add("green",2); 
    t.add("blue", 3); 
    t.dump();

    Index* idx = t.make_index("testindex");
    idx->dump();
 




}
