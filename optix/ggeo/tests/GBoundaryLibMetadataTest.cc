#include "GCache.hh"
#include "GBoundaryLibMetadata.hh"
#include "GBoundaryLib.hh"
#include "GBuffer.hh"
#include "GDomain.hh"
#include "stdio.h"
#include "string.h"



int main(int argc, char** argv)
{
    GCache gc("GGEOVIEW_"); 
    const char* idpath = gc.getIdPath(); 

    int wline = argc > 1 ? atoi(argv[1]) : -2 ;
 
    GBoundaryLibMetadata* meta = GBoundaryLibMetadata::load(idpath);
    meta->Summary(idpath);
    meta->dumpNames();

    unsigned int line = meta->getMaterialLine("GdDopedLS");
    printf(" GdLs : %u \n", line);


    GBuffer* wbuf = GBuffer::load<float>(idpath, "wavelength.npy");
    wbuf->Summary("wavelength buffer");

    GBuffer* obuf = GBuffer::load<unsigned int>(idpath, "optical.npy");
    // GBuffer* obuf = GBuffer::load<int>(idpath, "optical.npy");
    //GBuffer* obuf = GBuffer::load<float>(idpath, "optical.npy");
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
