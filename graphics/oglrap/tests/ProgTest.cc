#include "Prog.hh"
#include "stdlib.h"

/*
delta:oglrap blyth$ SHADER_DIR=~/env/graphics/ggeoview/gl /usr/local/env/graphics/oglrap/bin/ProgTest
[2015-04-16 19:49:06.156754] [0x000007fff7a78431] [info]    Prog::Prog found directory at /Users/blyth/env/graphics/ggeoview/gl/nrm

*/


int main(int argc, char** argv)
{
    Prog prog(getenv("SHADER_DIR"), "nrm");
    return 0 ;
}


