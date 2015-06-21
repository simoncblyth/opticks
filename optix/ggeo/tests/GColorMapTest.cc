#include "GColorMap.hh"
#include "stdlib.h"
#include "stdio.h"

int main(int argc, char** argv)
{
    char* idpath = getenv("IDPATH");
    GColorMap* cm = GColorMap::load(idpath, "_Materials.json") ; 
    cm->dump();


    return 0 ;
}
