#include "GColorMap.hh"
#include "stdlib.h"
#include "stdio.h"
#include "assert.h"

int main(int argc, char** argv)
{
    char* idpath = getenv("IDPATH");
    GColorMap* cm = GColorMap::load(idpath, "GMaterialIndexColors.json") ; 
    assert(cm && "FAILED TO LOAD COLORMAP");
    cm->dump();


    return 0 ;
}
