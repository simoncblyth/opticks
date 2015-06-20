#include "GSurfaceIndex.hh"
#include "stdio.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
       printf("%s : expecting path of directory which contains a json files \n", argv[0]);
       return 1 ; 
    }

    GSurfaceIndex* idx = GSurfaceIndex::load(argv[1]);
    //idx->dump();
    idx->test();

    return 0 ;
}
