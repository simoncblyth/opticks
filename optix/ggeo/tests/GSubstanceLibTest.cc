#include "GSubstanceLib.hh"
#include "stdio.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
       printf("%s : expecting path of directory which contains files named: wavelength.npy and GSubstanceLibMetadata.json\n", argv[0]);
       return 1 ; 
    }

    const char* dir = argv[1];
    GSubstanceLib* lib = GSubstanceLib::load(dir);
    lib->Summary("load ");


    if(argc == 2)
    {
        lib->dumpWavelengthBuffer();        
    }
    else
    {
        for(unsigned int i=2 ; i < argc ; i++) lib->dumpWavelengthBuffer(atoi(argv[i]));        
    }


    return 0 ;
}
