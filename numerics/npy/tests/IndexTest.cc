#include "Index.hpp"
#include "stdio.h"
#include "stdlib.h"
#include <iostream>

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    if(!idpath)
    {
        std::cout << argv[0] << " missing IDPATH " << std::endl ; 
        return 0 ; 
    } 

    Index* idx = Index::load(idpath, "GMaterialIndex");  // itemname => index

    idx->dumpPaths(idpath) ;

    idx->test();
    //idx->formTable();

    idx->setExt(".ini");
    idx->save("/tmp");

    return 0 ; 
}

