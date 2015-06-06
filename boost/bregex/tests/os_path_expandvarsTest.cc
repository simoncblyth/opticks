#include "regexsearch.hh"
#include "stdio.h"

int main(int argc, char** argv)
{
    std::string o = os_path_expandvars("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    printf("%s\n", o.c_str());
    return 0 ; 
}
