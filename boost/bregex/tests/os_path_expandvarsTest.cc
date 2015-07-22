#include "regexsearch.hh"
#include "stdio.h"

int main(int argc, char** argv)
{
    std::string o = os_path_expandvars("$ENV_HOME/graphics/ggeoview/cu/photon.h");
    printf("%s\n", o.c_str());
    std::string home = os_path_expandvars("$HOME/.opticks/GColors.json");
    printf("%s\n", home.c_str());
    std::string home1 = os_path_expandvars("$HOME/.opticks");
    printf("%s\n", home1.c_str());
    std::string home2 = os_path_expandvars("$HOME/");
    printf("%s\n", home2.c_str());
    std::string home3 = os_path_expandvars("$HOME");
    printf("%s\n", home3.c_str());

    std::string home4 = os_path_expandvars("$HOME/$ENV_HOME");
    printf("%s\n", home4.c_str());
    return 0 ; 
}
