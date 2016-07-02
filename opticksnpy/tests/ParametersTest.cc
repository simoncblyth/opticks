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


    const char* path = "/tmp/parameters.json" ;
    p.save(path);

    Parameters* q = Parameters::load(path);
    q->dump("q");
}


int main()
{
    test_basic();
    test_save_load();
    return 0 ; 
}
