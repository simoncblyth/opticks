#include "Index.hpp"
#include "stdio.h"
#include "stdlib.h"
#include <iostream>

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");
    const char* idxtype = "GMaterialIndex" ;

    if(!idpath)
    {
        std::cout << argv[0] << " missing IDPATH " << std::endl ; 
        return 0 ; 
    } 

    Index* idx = Index::load(idpath, idxtype );  // itemname => index
    if(!idx)
    {
        std::cout << argv[0] << " failed to load index " << idxtype << std::endl ; 
        return 0 ; 
    }

    idx->dumpPaths(idpath) ;

    idx->test();
    //idx->formTable();

    idx->setExt(".ini");
    idx->save("/tmp");

    return 0 ; 
}

