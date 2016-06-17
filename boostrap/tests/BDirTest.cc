#include "BFile.hh"
#include "BDir.hh"
#include <iostream>


typedef std::vector<std::string> VS ; 

void dump(const char* msg, std::vector<std::string>& names)
{
    std::cerr << msg << std::endl ; 
    for(VS::const_iterator it=names.begin() ; it != names.end() ; it++ ) std::cerr << *it << std::endl ; 
}


int main(int argc, char** argv)
{
    std::string home = BFile::FormPath("~");
    const char* dir = home.c_str();

    VS names ;
    BDir::dirlist(names, dir);
    dump("all", names);
     
    names.clear();
    BDir::dirlist(names, dir, ".ini" );
    dump(".ini", names);

    names.clear();
    BDir::dirlist(names, dir, ".json" );
    dump(".json", names);
  
}
