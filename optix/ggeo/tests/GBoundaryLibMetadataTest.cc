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
    int wline = argc > 2 ? atoi(argv[2]) : -2 ;
 
    GBoundaryLibMetadata* meta = GBoundaryLibMetadata::load(path);
    meta->Summary(path);
    meta->dumpNames();


    GBuffer* wbuf = GBuffer::load<float>(path, "wavelength.npy");
    wbuf->Summary("wavelength buffer");

    GBuffer* obuf = GBuffer::load<unsigned int>(path, "optical.npy");
    obuf->Summary("optical buffer");


    unsigned int numElementsTotal = wbuf->getNumElementsTotal();
    unsigned int numProp = GBoundaryLib::NUM_QUAD*4  ; 
    unsigned int domainLength = GBoundaryLib::DOMAIN_LENGTH ;
    unsigned int numBoundary = numElementsTotal/(numProp*domainLength);

    GDomain<float>* domain = GBoundaryLib::getDefaultDomain();
    assert(domain->getLength() == domainLength);

    if(wline > -2)
    {
         GBoundaryLib::dumpWavelengthBuffer(wline, wbuf, meta, numBoundary, domainLength );
         GBoundaryLib::dumpOpticalBuffer(   wline, obuf, meta, numBoundary );
    }


    return 0 ;
}
