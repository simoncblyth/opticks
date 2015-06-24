#include "Index.hpp"
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");

    Index* idx = Index::load(idpath, "GMaterialIndex");                      // itemname => index

    idx->test();
    //idx->formTable();

    return 0 ; 
}

