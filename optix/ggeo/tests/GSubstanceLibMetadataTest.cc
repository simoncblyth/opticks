#include "GSubstanceLibMetadata.hh"
#include "stdio.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
       printf("%s : expecting path of directory which contains a file named: GSubstanceLibMetadata.json\n", argv[0]);
       return 1 ; 
    }

    const char* path = argv[1];
    GSubstanceLibMetadata* meta = GSubstanceLibMetadata::load(path);
    //meta->Summary(path);
    meta->createMaterialMap();

    return 0 ;
}
