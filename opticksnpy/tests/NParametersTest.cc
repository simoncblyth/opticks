#include <cassert>
#include "NParameters.hpp"



void test_basic()
{
    NParameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);
    p.dump();

}

void test_save_load()
{
    NParameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);
    p.dump();


    const char* path = "$TMP/parameters.json" ;
    p.save(path);

    NParameters* q = NParameters::load(path);
    q->dump("q");
}


void test_set()
{
    NParameters p ;
    p.add<std::string>("red", "g");
    p.add<std::string>("green", "g");
    p.add<std::string>("blue", "b");
    p.dump();

    p.set<std::string>("red","r");
    p.set<std::string>("cyan","c");
    p.dump();
}

void test_bool_nonexisting()
{
    NParameters p ;

    bool non = p.get<bool>("NonExisting","0");
    assert(non == false); 
    bool oui = p.get<bool>("NonExisting","1");
    assert(oui == true); 
}
void test_bool()
{
    NParameters a ;
    a.add<bool>("Existing",true);
    bool yes = a.get<bool>("Existing","0");
    assert(yes == true); 

    NParameters b ;
    b.add<bool>("Existing",false);
    bool no1 = b.get<bool>("Existing","0");
    assert(no1 == false); 
    bool no2 = b.get<bool>("Existing","1");
    assert(no2 == false); 


}


int main()
{
    test_basic();
    test_save_load();
    test_set();
    test_bool_nonexisting();
    test_bool();

    return 0 ; 
}
