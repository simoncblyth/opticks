#include "GColors.hh"
#include "stdlib.h"

int main(int argc, char** argv)
{
    GColors* gc = GColors::load("$HOME/.opticks","GColors.json");
    gc->dump();
    gc->test(); 



}
