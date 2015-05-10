#include "GSubstanceLibMetadata.hh"
#include "GSubstanceLib.hh"
#include "GBuffer.hh"
#include "GDomain.hh"
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

    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    unsigned int numProp = 16 ; 
    unsigned int domainLength = GSubstanceLib::DOMAIN_LENGTH ;
    unsigned int numSubstance = numElementsTotal/(numProp*domainLength);

    GDomain<float>* domain = GSubstanceLib::getDefaultDomain();
    assert(domain->getLength() == domainLength);

    int wline = -1; 
    GSubstanceLib::dumpWavelengthBuffer(wline, buffer, meta, numSubstance, numProp, domainLength );

    return 0 ;
}
