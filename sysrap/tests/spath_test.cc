// name=spath_test ; gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name && lldb__ /tmp/$name

#include <iostream>
#include "spath.h"

int main(int argc, char** argv)
{
    std::vector<std::string> specs = {"$HOME/hello.npy", "$HOME" } ; 

    for(unsigned i=0 ; i < specs.size() ; i++)
    { 
        const char* spec = specs[i].c_str(); 
        const char* path = spath::Resolve(spec); 
        std::cout 
            << " spec " << spec << std::endl  
            << " path " << path << std::endl
            << std::endl 
            ;
    }

    return 0 ; 
}
