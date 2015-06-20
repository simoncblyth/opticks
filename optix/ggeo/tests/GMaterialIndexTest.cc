#include "GMaterialIndex.hh"
#include "stdio.h"
#include "string.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
       printf("%s : expecting path of directory which contains a file named: %s \n", argv[0], GMaterialIndex::LOCAL_NAME);
       return 1 ; 
    }

    const char* path = argv[1];
    GMaterialIndex* idx = GMaterialIndex::load(path);
    idx->dump();
    idx->test();


    return 0 ;
}
