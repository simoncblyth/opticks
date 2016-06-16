#include "BDir.hh"
#include <cstdio>

int main(int argc, char** argv)
{
    std::vector<std::string> names ;
    printf("all\n");
    BDir::dirlist(names, "/Users/blyth/.opticks/rainbow/State");
    printf(".ini\n");
    BDir::dirlist(names, "/Users/blyth/.opticks/rainbow/State", ".ini" );
    printf(".json\n");
    BDir::dirlist(names, "/Users/blyth/.opticks/rainbow/State", ".json" );
  
}
