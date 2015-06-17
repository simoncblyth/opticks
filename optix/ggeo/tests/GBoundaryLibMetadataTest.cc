#include "GBoundaryLibMetadata.hh"
#include "GBoundaryLib.hh"
#include "GBuffer.hh"
#include "GDomain.hh"
#include "stdio.h"
#include "string.h"

int main(int argc, char** argv)
{
    if(argc < 2)
    {
       printf("%s : expecting path of directory which contains a file named: GBoundaryLibMetadata.json\n", argv[0]);
       return 1 ; 
    }

    const char* path = argv[1];
    GBoundaryLibMetadata* meta = GBoundaryLibMetadata::load(path);
    meta->Summary(path);
    meta->dumpNames();


    char wpath[256];
    snprintf(wpath, 256,"%s/wavelength.npy", path); 
    GBuffer* buffer = GBuffer::load<float>(wpath);
    buffer->Summary("wavelength buffer");

    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    unsigned int numProp = GBoundaryLib::NUM_QUAD*4  ; 
    unsigned int domainLength = GBoundaryLib::DOMAIN_LENGTH ;
    unsigned int numBoundary = numElementsTotal/(numProp*domainLength);

    GDomain<float>* domain = GBoundaryLib::getDefaultDomain();
    assert(domain->getLength() == domainLength);

    //int wline = -1; 
    //GBoundaryLib::dumpWavelengthBuffer(wline, buffer, meta, numBoundary, domainLength );

    return 0 ;
}
