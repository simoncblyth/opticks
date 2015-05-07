#include "GSubstanceLibMetadata.hh"
#include "GBuffer.hh"
#include "stdio.h"
#include "string.h"

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

    char wpath[256];
    snprintf(wpath, 256,"%s/wavelength.npy", path); 
    GBuffer* buffer = GBuffer::load<float>(wpath);
    buffer->Summary("wavelength buffer");


    return 0 ;
}
