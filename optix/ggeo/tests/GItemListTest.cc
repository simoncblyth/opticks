#include "GItemList.hh"

#include <string>
#include "Map.hpp"


int main(int argc, char** argv)
{
    GItemList l("testlist");

    l.add("red");
    l.add("green");
    l.add("blue");
    l.add("cyan");
    l.add("magenta");
    l.add("yellow");
    l.dump();

    l.save("/tmp");

    GItemList* t = GItemList::load("/tmp", "testlist");
    if(t) t->dump();

    Map<std::string, unsigned int>* m = new Map<std::string, unsigned int>() ;

    m->add("yellow",1);
    m->add("magenta",2);
    m->add("cyan",3);

    m->dump("sort order");

    t->setOrder(m->getMap());
    t->sort();
    t->dump("after sort");


    return 0 ;

}
