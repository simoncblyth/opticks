#define GUI_ 1
#include "GMaterialIndex.hh"
#include "GColors.hh"
#include "GColorMap.hh"
#include "GBuffer.hh"

#include "stdio.h"
#include "stdlib.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");

    GMaterialIndex* idx = GMaterialIndex::load(idpath);                      // itemname => index

    GColorMap* cmap = GColorMap::load(idpath, "GMaterialIndexColors.json");  // itemname => colorname 
    idx->setColorMap(cmap);

    GColors* colors = GColors::load(idpath,"GColors.json");           // colorname => hexcode 
    idx->setColorSource(colors);

    idx->test();
    idx->dump();

    //GBuffer* buffer = idx->getColorBuffer();
    //printf("makeColorBuffer %u \n", buffer->getNumBytes() );
    //colors->dump_uchar4_buffer(buffer);

    idx->formTable();
    idx->gui();

    return 0 ;
}
