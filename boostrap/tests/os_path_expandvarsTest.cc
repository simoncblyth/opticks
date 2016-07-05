#include "regexsearch.hh"
#include "fsutil.hh"
#include <cstdio>
#include <vector>

int main(int argc, char** argv)
{
    printf("%s\n", argv[0]);

    // observe special casing of getenv("HOME") on mingw 
    // so better to avoid HOME ? 

    std::vector<std::string> ss ; 
    ss.push_back("$OPTICKS_HOME/optickscore/OpticksPhoton.h");
    ss.push_back("$HOME/.opticks/GColors.json");
    ss.push_back("$HOME/.opticks");
    ss.push_back("$HOME/");
    ss.push_back("$HOME");
    ss.push_back("$OPTICKS_HOME");
    ss.push_back("$HOME/$OPTICKS_HOME");

    for(unsigned int i=0 ; i < ss.size() ; i++)
    {

       std::string s = ss[i] ;
       //std::string x = s ;
       //std::string x = os_path_expandvars(s.c_str() );
       std::string x = fsutil::FormPath(s.c_str() );
       printf("  [%s] -->  [%s] \n", s.c_str(), x.c_str());
    }


    return 0 ; 
}
