#include <cstdio>
#include "DynamicDefine.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    printf("%s\n", argv[0]);

    DynamicDefine dd ;
    dd.add<unsigned int>("MAXREC", 10);
    dd.add<float>("OTHER", 20.0);

    printf("%s write...\n", argv[0]);
    dd.write("/tmp", "DynamicDefineTest.h");

    return 0 ; 
}
