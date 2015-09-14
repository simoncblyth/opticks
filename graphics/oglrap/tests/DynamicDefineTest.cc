#include "DynamicDefine.hh"

int main()
{
    DynamicDefine dd ;
    dd.add<unsigned int>("MAXREC", 10);
    dd.add<float>("OTHER", 20.0);
    dd.write("/tmp", "DynamicDefineTest.h");

    return 0 ; 
}
