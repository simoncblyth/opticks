#include "GColors.hh"
#include "stdlib.h"

int main(int argc, char** argv)
{
    //const char* idpath = getenv("IDPATH") ;
    //GColors* gc = GColors::load(idpath);
    GColors* gc = GColors::load("$HOME/.opticks","GColors.json");
    gc->dump();
    gc->test(); 



}
