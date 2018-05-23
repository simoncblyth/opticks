#include "NTris.hpp"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    NTris* tris = NTris::make_sphere();
    tris->dump();

    return 0 ; 
}



