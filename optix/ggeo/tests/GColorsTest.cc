#include "GColors.hh"
#include "stdlib.h"

int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH") ;
    GColors* gc = GColors::load(idpath);
    gc->dump();
    gc->test(); 


}
