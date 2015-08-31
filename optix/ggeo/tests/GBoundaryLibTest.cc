#include "GCache.hh"
#include "GBoundaryLib.hh"
#include "GBoundary.hh"
#include "stdlib.h"
#include "stdio.h"

int main(int argc, char** argv)
{
    GCache gc("GGEOVIEW_");

    GBoundaryLib* lib = GBoundaryLib::load(gc.getIdPath());
    //lib->Summary("GBoundaryLib::Summary");

    for(unsigned int ib=0 ; ib < lib->getNumBoundary() ; ib++)
    {
        GBoundary* boundary = lib->getBoundary(ib);
        boundary->Summary("boundary");
    }


    if(argc > 1)
    {
        for(unsigned int i=1 ; i < argc ; i++) 
        {
            int b = atoi(argv[i]) ;
            printf("\nGBoundaryLib.dumpWavelengthBuffer %2d \n", b);
            lib->dumpWavelengthBuffer(b);        
        } 
    }

    return 0 ;
}
