#include "GBoundaryLib.hh"
#include "GBoundary.hh"
#include "stdlib.h"
#include "stdio.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
       printf("%s : expecting path of directory which contains files named: wavelength.npy and GBoundaryLibMetadata.json\n", argv[0]);
       return 1 ; 
    }

    const char* dir = argv[1];
    GBoundaryLib* lib = GBoundaryLib::load(dir);
    //lib->Summary("GBoundaryLib::Summary");

    for(unsigned int ib=0 ; ib < lib->getNumBoundary() ; ib++)
    {
        GBoundary* boundary = lib->getBoundary(ib);
        boundary->Summary("boundary");
    }

    /*
    if(argc == 2) lib->dumpWavelengthBuffer();        
    else          for(unsigned int i=2 ; i < argc ; i++) lib->dumpWavelengthBuffer(atoi(argv[i]));        
    */

    return 0 ;
}
