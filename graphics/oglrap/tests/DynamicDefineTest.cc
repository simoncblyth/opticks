#include "DynamicDefine.hh"

int main()
{
    DynamicDefine dd("/tmp", "DynamicDefineTest.h");
    dd.add<unsigned int>("MAXREC", 10);
    dd.add<float>("OTHER", 20.0);
    dd.write();

    return 0 ; 
}
