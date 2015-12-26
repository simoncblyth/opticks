#include "Parameters.hpp"


void test_basic()
{
    Parameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);
    p.dump();

}



int main()
{
    Parameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);
    p.dump("p");

    const char* path = "/tmp/parameters.json" ;
    p.save(path);

    Parameters* q = Parameters::load(path);
    q->dump("q");



    return 0 ; 
}
