// clang++ main.cc -o /tmp/loadppm

#define LOADPPM_IMPLEMENTATION
#include "loadPPM.h"

int main(int argc, char** argv)
{
    if(argc < 2) 
    {
        printf("%s : expecting argument with path to ppm file\n", argv[0]);
        return 1; 
    }
    char* path = argv[1] ;
    int width, height ;
    unsigned char* img = loadPPM(path, &width, &height);
    if(img)
    {
        printf("loaded %s of width %d height %d \n", path, width, height);
    }
    else
    {
        printf("failed to load %s  \n", path);
    }


    return 0 ;
}


