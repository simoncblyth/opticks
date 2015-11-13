// ggv --itemlist

#include "GCache.hh"
#include "GItemList.hh"

#include <string>
#include "Map.hpp"


void base()
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
}


void test_replaceFields(GCache* cache)
{
    GItemList* il = GItemList::load(cache->getIdPath(), "GPmt", "GPmt/0");
    il->dump();
    il->dumpFields();
    il->replaceField(0, "OUTERMATERIAL", "MineralOil" );
    il->replaceField(1, "OUTERSURFACE", "lvPmtHemiCathodeSensorSurface" );
    il->dumpFields("after replace");
}


void test_makeSlice(GCache* cache)
{
    GItemList* il = GItemList::load(cache->getIdPath(), "GPmt", "GPmt/0");
    il->dump();

    GItemList* sl = il->make_slice("0:4");
    sl->dump("sliced");
}








int main(int argc, char** argv)
{
    GCache* cache = new GCache("GGEOVIEW_", "GItemList.log", "info");

    //test_replaceFields(cache);
    test_makeSlice(cache);

    return 0 ;

}
