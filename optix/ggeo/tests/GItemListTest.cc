#include "GItemList.hh"

int main(int argc, char** argv)
{
    GItemList l("testlist");

    l.add("red");
    l.add("green");
    l.add("blue");
    l.dump();

    l.save("/tmp");


    GItemList* t = GItemList::load("/tmp", "testlist");
    if(t) t->dump();

    return 0 ;

}
