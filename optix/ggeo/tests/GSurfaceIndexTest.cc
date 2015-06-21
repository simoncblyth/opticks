#include "GSurfaceIndex.hh"
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    GSurfaceIndex* idx = GSurfaceIndex::load(idpath);
    //idx->dump();
    idx->test();

    return 0 ;
}
