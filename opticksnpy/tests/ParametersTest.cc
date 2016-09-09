#include <cassert>
#include "Parameters.hpp"



void test_basic()
{
    Parameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);
    p.dump();

}

void test_save_load()
{
    Parameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);
    p.dump();


    const char* path = "$TMP/parameters.json" ;
    p.save(path);

    Parameters* q = Parameters::load(path);
    q->dump("q");
}


void test_set()
{
    Parameters p ;
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
    Parameters p ;

    bool non = p.get<bool>("NonExisting","0");
    assert(non == false); 
    bool oui = p.get<bool>("NonExisting","1");
    assert(oui == true); 
}
void test_bool()
{
    Parameters a ;
    a.add<bool>("Existing",true);
    bool yes = a.get<bool>("Existing","0");
    assert(yes == true); 

    Parameters b ;
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
