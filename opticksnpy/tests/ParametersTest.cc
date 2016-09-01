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


int main()
{
    test_basic();
    test_save_load();
    test_set();
    return 0 ; 
}
