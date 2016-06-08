#include "dirutil.hh"
#include <cstdio>

int main(int argc, char** argv)
{
    std::vector<std::string> names ;
    printf("all\n");
    dirlist(names, "/Users/blyth/.opticks/rainbow/State");
    printf(".ini\n");
    dirlist(names, "/Users/blyth/.opticks/rainbow/State", ".ini" );
    printf(".json\n");
    dirlist(names, "/Users/blyth/.opticks/rainbow/State", ".json" );
  
}
