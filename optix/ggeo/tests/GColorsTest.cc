#include "GColors.hh"

int main(int argc, char** argv)
{
    const char* path = argv[1] ;
    GColors* gc = GColors::load(path);
    //gc->dump();
    gc->test(); 


}
